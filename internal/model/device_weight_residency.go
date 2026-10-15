package model

import (
	"reflect"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// modelHALWeightResidency owns immutable device weight handles for one loaded model.
// Request state does not enter this object: KV, recurrent/GDN state, activations and
// sampler/session bookkeeping remain owned and freed by Session.
type modelHALWeightResidency struct {
	mu       sync.Mutex
	backends []*backendHALWeightResidency
	closed   bool
}

// backendHALWeightResidency is one exact backend instance's immutable weight set.
// Backend names are deliberately not identity: two CUDA objects with the same name may
// address different devices or contexts, and sharing a handle across them is unsafe.
type backendHALWeightResidency struct {
	mu      sync.Mutex
	be      compute.Backend
	weights map[string]compute.Tensor
	closed  bool
}

// backendHasComparableIdentity reports whether interface equality is a safe object-
// identity check for this backend. A value backend can compare equal to a distinct copy,
// so only pointers may share; every other shape stays on the session-local path instead
// of guessing identity from a name, tier, or copied value.
func backendHasComparableIdentity(be compute.Backend) bool {
	if be == nil {
		return false
	}
	t := reflect.TypeOf(be)
	return t != nil && t.Kind() == reflect.Pointer && t.Comparable()
}

func (r *modelHALWeightResidency) backend(be compute.Backend) *backendHALWeightResidency {
	if r == nil || !backendHasComparableIdentity(be) {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	for _, resident := range r.backends {
		if resident.be == be {
			return resident
		}
	}
	resident := &backendHALWeightResidency{be: be, weights: make(map[string]compute.Tensor)}
	r.backends = append(r.backends, resident)
	return resident
}

func (r *backendHALWeightResidency) getOrStage(key string, stage func() compute.Tensor) compute.Tensor {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		panic("model: immutable device weight residency is closed")
	}
	if resident, ok := r.weights[key]; ok {
		return resident
	}
	resident := stage()
	r.weights[key] = resident
	return resident
}

func (r *modelHALWeightResidency) close() {
	if r == nil {
		return
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	backends := append([]*backendHALWeightResidency(nil), r.backends...)
	r.backends = nil
	r.mu.Unlock()

	for _, resident := range backends {
		resident.close()
	}
}

func (resident *backendHALWeightResidency) close() {
	scope := beginBackendExecution(resident.be)
	complete := false
	defer scope.finishCompleted(&complete)
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if !resident.closed {
		resident.closed = true
		for key, weight := range resident.weights {
			resident.be.Free(weight)
			delete(resident.weights, key)
		}
	}
	// A failed Free leaves retirement uncertain. Preserve its unwind, including
	// legacy panic(nil), rather than allowing the checkpoint owner to unmap.
	complete = true
}

// immutableDeviceWeights returns model-lifetime residency only for a genuine device
// backend. The reference CPU's Upload is a host identity operation and keeps the existing
// session-local memoizer/lifetime unchanged.
func (m *Model) immutableDeviceWeights(be compute.Backend) *backendHALWeightResidency {
	if m == nil || be == nil || !be.Caps().DeviceMemory || !backendHasComparableIdentity(be) {
		return nil
	}
	closer := m.ensureWeightCloser()
	closer.mu.Lock()
	defer closer.mu.Unlock()
	// A session admitted before CloseWeights set closing must still find the pool it
	// borrowed from so Session.Close does not free model-owned handles. Only creation of
	// a new pool is refused after teardown starts.
	if closer.halWeights != nil {
		return closer.halWeights.backend(be)
	}
	if closer.closing || closer.closed {
		return nil
	}
	closer.halWeights = &modelHALWeightResidency{}
	return closer.halWeights.backend(be)
}

// lookupImmutableWeight adopts only an existing immutable handle. It never creates
// owner state, stages storage or calls a backend. This lets callers validate every
// missing host payload before the first upload, without rereading resident weights.
// The caller serializes session use, just as for cachedImmutableWeight.
func (s *Session) lookupImmutableWeight(sessionKey, residencyKey string) (compute.Tensor, bool) {
	if s == nil || s.halClosed {
		return compute.Tensor{}, false
	}
	if weight, ok := s.halW[sessionKey]; ok {
		return weight, true // Preserve the local handle's existing ownership marker.
	}
	// Model adoption needs both the lifetime pin and a local memo: resident chains
	// collect protected operands from halW before executing any weight closures.
	if s.M == nil || !s.modelWeightsHeld || s.halW == nil || !backendHasComparableIdentity(s.Backend) {
		return compute.Tensor{}, false
	}
	weightCloserInitMu.Lock()
	closer := s.M.weightCloser
	weightCloserInitMu.Unlock()
	if closer == nil {
		return compute.Tensor{}, false
	}
	closer.mu.Lock()
	defer closer.mu.Unlock()
	// A pinned session may still borrow while CloseWeights waits for it. The
	// final release cannot retire these handles until this session detaches.
	if closer.closed || closer.halWeights == nil {
		return compute.Tensor{}, false
	}
	registry := closer.halWeights
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.closed {
		return compute.Tensor{}, false
	}
	for _, resident := range registry.backends {
		if resident.be != s.Backend {
			continue
		}
		resident.mu.Lock()
		defer resident.mu.Unlock()
		if resident.closed {
			return compute.Tensor{}, false
		}
		weight, ok := resident.weights[residencyKey]
		if !ok {
			return compute.Tensor{}, false
		}
		s.halW[sessionKey] = weight
		if s.borrowedHALW == nil {
			s.borrowedHALW = make(map[string]struct{})
		}
		s.borrowedHALW[sessionKey] = struct{}{}
		return weight, true
	}
	return compute.Tensor{}, false
}

// VisitImmutableDeviceWeights observes already staged model-owned weights for one
// exact backend instance. It never creates a residency pool or stages a weight.
// The caller must invoke it outside token loops. false means unavailable (closed
// owner, unsupported identity, or rejected observation); an open, absent pool is
// known-empty. Session-local tensors, expert rings, KV and backend caches are excluded.
//
// visit runs with the model closer, residency registry and backend-weight locks
// held, so each tensor remains live throughout the observation. It must only read
// backend metadata: it must not retain tensors, free/stage storage, reenter Model
// or Session methods, or acquire a lock held while entering model code. A false
// return stops the traversal; callers must discard any partial inventory.
func (m *Model) VisitImmutableDeviceWeights(be compute.Backend, visit func(compute.Tensor) bool) bool {
	if m == nil || !backendHasComparableIdentity(be) || visit == nil {
		return false
	}
	// Read the existing closer without ensureWeightCloser: observation must not
	// create owner state. Initialization uses this same mutex.
	weightCloserInitMu.Lock()
	closer := m.weightCloser
	weightCloserInitMu.Unlock()
	if closer == nil {
		return true
	}
	closer.mu.Lock()
	defer closer.mu.Unlock()
	if closer.closing || closer.closed {
		return false
	}
	registry := closer.halWeights
	if registry == nil {
		return true
	}
	registry.mu.Lock()
	defer registry.mu.Unlock()
	if registry.closed {
		return false
	}
	for _, resident := range registry.backends {
		if resident.be != be {
			continue
		}
		resident.mu.Lock()
		defer resident.mu.Unlock()
		if resident.closed {
			return false
		}
		for _, tensor := range resident.weights {
			if !visit(tensor) {
				return false
			}
		}
		break
	}
	return true
}
