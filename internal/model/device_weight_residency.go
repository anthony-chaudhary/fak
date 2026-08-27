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
		resident.mu.Lock()
		if !resident.closed {
			resident.closed = true
			for key, weight := range resident.weights {
				resident.be.Free(weight)
				delete(resident.weights, key)
			}
		}
		resident.mu.Unlock()
	}
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
