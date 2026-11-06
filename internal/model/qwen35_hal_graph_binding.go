package model

import (
	"reflect"
	"slices"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// graphInheritedFailureLocked propagates graph-only invalidity across a COW
// clone. The containing state lock is held; a replacement owner cannot erase a
// failed mutation of the physical pair from which it was copied.
func (l *qwen35HALLayerState) graphInheritedFailureLocked() bool {
	if l.owner == nil {
		return false
	}
	l.owner.mu.Lock()
	defer l.owner.mu.Unlock()
	return l.owner.graphInvalid || l.owner.graphMutating
}

// An epoch counts model-owned mutation attempts, not tokens or successful GPU
// completions. Every ordinary decode/sequence callback advances it before work.
// Eager execution retains its existing failure policy; only graph admission is
// made sticky-false here. Replacing a physical owner starts a distinct identity.
func (o *qwen35HALLayerOwner) graphMutationBeginLocked() {
	if o.graphMutating || o.graphEpoch == ^uint64(0) {
		o.graphInvalid = true
	} else {
		o.graphEpoch++
	}
	o.graphMutating = true
}

func (o *qwen35HALLayerOwner) graphMutationFinishLocked(normalSuccess bool) {
	if !normalSuccess {
		o.graphInvalid = true
	}
	o.graphMutating = false
}

// qwen35GraphBinding is an immutable selection receipt, not a graph owner or a
// physical allocation lease. It retains Go identity/metadata without increasing
// recurrent refs (which would force COW). Only the session owns physical state.
// No serving route consumes this receipt yet; hybrid capture-once stays excluded.
// A future native replay transaction must account for its own epoch transition,
// checked completion and retirement before it can consume this model gate.
type qwen35GraphBinding struct {
	session  *Session
	model    *Model
	backend  compute.Backend
	domain   *sync.Mutex
	state    *qwen35HALState
	kv       compute.KVStore
	kvRows   int
	linear   []bool
	geometry [5]int
	layers   []qwen35GraphLayerBinding
	invalid  bool
}

type qwen35GraphLayerBinding struct {
	layer           int
	owner           *qwen35HALLayerOwner
	epoch           uint64
	conv, recurrent qwen35GraphTensorBinding
}

type qwen35GraphTensorBinding struct {
	buffer  compute.Buffer
	backend compute.Backend
	shape   []int
}

// Graph targets are pointer-backed identities. Refuse noncomparable or typed-nil
// implementations rather than panic in interface equality or infer value identity.
func qwen35GraphSamePointer(a, b any) bool {
	if a == nil || b == nil {
		return false
	}
	av, bv := reflect.ValueOf(a), reflect.ValueOf(b)
	return av.Type() == bv.Type() && av.Kind() == reflect.Pointer && !av.IsNil() && !bv.IsNil() && a == b
}

func qwen35GraphTensorMatches(t compute.Tensor, b qwen35GraphTensorBinding) bool {
	return t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil &&
		qwen35GraphSamePointer(t.Buf(), b.buffer) && qwen35GraphSamePointer(t.Backend(), b.backend) &&
		slices.Equal(t.Shape, b.shape) && t.Ready()
}

func qwen35GraphTensorBind(t compute.Tensor, be compute.Backend, shape []int) (qwen35GraphTensorBinding, bool) {
	b := qwen35GraphTensorBinding{buffer: t.Buf(), backend: be, shape: append([]int(nil), shape...)}
	return b, qwen35GraphTensorMatches(t, b)
}

func qwen35GraphGeometry(cfg Config) [5]int {
	return [5]int{cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim, cfg.LinearConvKernelDim}
}

// Consult the already-held public recording scope; never acquire its
// non-reentrant mutex. A receipt can be checked in a later operation's live scope
// only when that scope owns the same domain and is bound to the same session.
func qwen35GraphLiveDomain(s *Session) *sync.Mutex {
	if s == nil || s.M == nil || s.halClosed || s.halFailure != nil || !qwen35GraphSamePointer(s.Backend, s.Backend) {
		return nil
	}
	// Reject typed-nil and non-pointer receivers before invoking a capability;
	// a nonnil interface alone does not establish a safe method receiver.
	scope := s.backendExecutionScope
	if scope == nil || !scope.live || scope.owner == nil {
		return nil
	}
	provider, ok := s.Backend.(compute.ForwardExecutionOwnerBackend)
	if !ok || provider.ForwardExecutionOwner() != scope.owner {
		return nil
	}
	return scope.owner
}

// bindQwen35GraphOwned observes private, ready physical state under the live
// public operation scope. Missing owners, incomplete layer inventories and
// mismatched F32 state geometry refuse binding without adopting or cloning data.
func (s *Session) bindQwen35GraphOwned() (*qwen35GraphBinding, bool) {
	domain := qwen35GraphLiveDomain(s)
	if domain == nil || !s.M.Cfg.IsQwen35Hybrid() || s.M.Cfg.NumLayers <= 0 || s.qwen35HAL == nil ||
		!qwen35GraphSamePointer(s.halKV, s.halKV) || !qwen35GraphSamePointer(s.Backend, s.Backend) {
		return nil, false
	}
	cfg := s.M.Cfg
	layout, err := ComputeQwen35StateLayout(cfg, "f32")
	if err != nil || layout.ConvElements == 0 {
		return nil, false
	}
	q := s.qwen35HAL
	kvRows := s.halKV.Len()
	if kvRows < 0 {
		return nil, false
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.sequenceFailure != nil || !qwen35GraphSamePointer(q.backend, s.Backend) || len(q.layers) != cfg.NumLayers {
		return nil, false
	}
	b := &qwen35GraphBinding{session: s, model: s.M, backend: s.Backend, domain: domain, state: q, kv: s.halKV, kvRows: kvRows,
		geometry: qwen35GraphGeometry(cfg), linear: make([]bool, cfg.NumLayers)}
	for l := range b.linear {
		b.linear[l] = cfg.isLinearAttnLayer(l)
		if !b.linear[l] {
			continue
		}
		st := &q.layers[l]
		owner := st.owner
		if owner == nil {
			return nil, false
		}
		entry, ok := func() (qwen35GraphLayerBinding, bool) {
			owner.mu.Lock()
			defer owner.mu.Unlock()
			if owner.refs != 1 || owner.graphInvalid || owner.graphMutating || !qwen35GraphSamePointer(owner.backend, s.Backend) {
				return qwen35GraphLayerBinding{}, false
			}
			conv, cok := qwen35GraphTensorBind(owner.conv, s.Backend, []int{layout.ConvRows, layout.ConvDim})
			recurrent, rok := qwen35GraphTensorBind(owner.recurrent, s.Backend, []int{layout.RecurrentHeads, layout.KeyHeadDim, layout.ValueHeadDim})
			if !cok || !rok || !qwen35GraphTensorMatches(st.conv, conv) || !qwen35GraphTensorMatches(st.recurrent, recurrent) {
				return qwen35GraphLayerBinding{}, false
			}
			return qwen35GraphLayerBinding{layer: l, owner: owner, epoch: owner.graphEpoch, conv: conv, recurrent: recurrent}, true
		}()
		if !ok || qwen35GraphSamePointer(entry.conv.buffer, entry.recurrent.buffer) {
			return nil, false
		}
		// The model receipt excludes repeated physical handles across all layer roles;
		// native pointer/range alias checks remain a separate backend obligation.
		for _, old := range b.layers {
			if old.owner == entry.owner || qwen35GraphSamePointer(old.conv.buffer, entry.conv.buffer) ||
				qwen35GraphSamePointer(old.conv.buffer, entry.recurrent.buffer) || qwen35GraphSamePointer(old.recurrent.buffer, entry.conv.buffer) ||
				qwen35GraphSamePointer(old.recurrent.buffer, entry.recurrent.buffer) {
				return nil, false
			}
		}
		b.layers = append(b.layers, entry)
	}
	if len(b.layers) == 0 {
		return nil, false
	}
	return b, true
}

// matchesOwned never refreshes epochs, owner selection or tensor descriptors.
// An observed mismatch under the live domain permanently invalidates this receipt.
// Expired/missing scopes cannot validate or mutate it. Normal in-place mutation is
// expected to make a receipt stale; a new binding is required, not content equality.
func (b *qwen35GraphBinding) matchesOwned(s *Session) bool {
	domain := qwen35GraphLiveDomain(s)
	if b == nil || domain == nil || domain != b.domain {
		return false
	}
	valid := false
	defer func() {
		if !valid {
			b.invalid = true
		}
	}()
	if b.invalid || s != b.session || s.M != b.model || s.qwen35HAL != b.state ||
		!qwen35GraphSamePointer(s.Backend, b.backend) || !qwen35GraphSamePointer(s.halKV, b.kv) || s.halKV.Len() != b.kvRows {
		return false
	}
	cfg := s.M.Cfg
	if !cfg.IsQwen35Hybrid() || cfg.NumLayers != len(b.linear) || qwen35GraphGeometry(cfg) != b.geometry {
		return false
	}
	for l, linear := range b.linear {
		if cfg.isLinearAttnLayer(l) != linear {
			return false
		}
	}
	q := b.state
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.sequenceFailure != nil || !qwen35GraphSamePointer(q.backend, b.backend) || len(q.layers) != len(b.linear) {
		return false
	}
	count := 0
	for _, linear := range b.linear {
		if linear {
			count++
		}
	}
	if count != len(b.layers) {
		return false
	}
	at := 0
	for l, linear := range b.linear {
		if !linear {
			continue
		}
		entry := &b.layers[at]
		at++
		if entry.layer != l || q.layers[l].owner != entry.owner || entry.owner == nil {
			return false
		}
		ok := func() bool {
			owner := entry.owner
			owner.mu.Lock()
			defer owner.mu.Unlock()
			return owner.refs == 1 && !owner.graphInvalid && !owner.graphMutating && owner.graphEpoch == entry.epoch &&
				qwen35GraphSamePointer(owner.backend, b.backend) && qwen35GraphTensorMatches(owner.conv, entry.conv) &&
				qwen35GraphTensorMatches(owner.recurrent, entry.recurrent) && qwen35GraphTensorMatches(q.layers[l].conv, entry.conv) &&
				qwen35GraphTensorMatches(q.layers[l].recurrent, entry.recurrent)
		}()
		if !ok {
			return false
		}
	}
	valid = true
	return true
}
