package model

import (
	"errors"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_model_scope.go — the serve owner of R7's SharedExpertRing (#5618). A serve builds a
// session per request, and a session-scoped ring is built lazily and freed at Close, so every
// request paged its routed experts in again from the checkpoint. EnableSharedExpertRing gives the
// (Model, Backend) pair one ring sized to the checkpoint tier's declared device budget; every
// session on that backend attaches at construction and detaches at Close, so residency outlives
// the request and concurrent sessions share one budget instead of overcommitting it.

// ErrSharedRingUndeclared means the model declares no checkpoint device ring budget, so there is
// no bounded routed-expert residency to share.
var ErrSharedRingUndeclared = errors.New("model: no declared routed-expert device ring budget to share")

var (
	sharedExpertRingsMu   sync.Mutex
	sharedExpertRingAgent atomic.Uint64
)

// EnableSharedExpertRing builds, or returns, the model-scoped routed-expert ring for be. Its budget
// is the checkpoint tier's declared device ring budget, which is the budget every session already
// validates against. Sessions created on be afterwards attach to it; it is freed by CloseWeights.
func (m *Model) EnableSharedExpertRing(be compute.Backend) (*SharedExpertRing, error) {
	return m.EnableSharedExpertRingWith(be, SharedExpertRingOptions{})
}

// SharedExpertRingOptions are the construction-time ring knobs a serve may set; the zero value is
// the plain LRU ring EnableSharedExpertRing has always built.
type SharedExpertRingOptions struct {
	Evict ExpertRingEvictPolicy
	// StableBytes is the scan-resistant stable-set ceiling (expert_ring_stable.go); 0 = off.
	StableBytes int64
}

// EnableSharedExpertRingWith is EnableSharedExpertRing with construction-time options (fixed for
// the ring's life). A ring that already exists for be is returned unchanged.
func (m *Model) EnableSharedExpertRingWith(be compute.Backend, opts SharedExpertRingOptions) (*SharedExpertRing, error) {
	if m == nil {
		return nil, ErrSharedRingModel
	}
	if be == nil || !reflect.TypeOf(be).Comparable() {
		return nil, ErrSharedRingBackend
	}
	budget := m.expertCheckpoint.DeviceRingBudget()
	if budget <= 0 {
		return nil, ErrSharedRingUndeclared
	}
	sharedExpertRingsMu.Lock()
	defer sharedExpertRingsMu.Unlock()
	if sh := m.sharedExpertRings[be]; sh != nil {
		return sh, nil
	}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget, Evict: opts.Evict, StableBytes: opts.StableBytes})
	if err != nil {
		return nil, err
	}
	if m.sharedExpertRings == nil {
		m.sharedExpertRings = map[compute.Backend]*SharedExpertRing{}
	}
	m.sharedExpertRings[be] = sh
	return sh, nil
}

// SharedExpertRingFor returns the model-scoped ring for be, or nil when none is enabled.
func (m *Model) SharedExpertRingFor(be compute.Backend) *SharedExpertRing {
	if m == nil || be == nil {
		return nil
	}
	sharedExpertRingsMu.Lock()
	defer sharedExpertRingsMu.Unlock()
	// A backend whose dynamic type is not comparable cannot key the registry (and cannot have
	// been enabled), so it never attaches.
	if len(m.sharedExpertRings) == 0 || !reflect.TypeOf(be).Comparable() {
		return nil
	}
	return m.sharedExpertRings[be]
}

// attachSharedExpertRing points a new session at the model-scoped ring for its backend. A model
// without one leaves the session's lazily built per-session ring path unchanged.
func (m *Model) attachSharedExpertRing(s *Session) error {
	sh := m.SharedExpertRingFor(s.Backend)
	if sh == nil {
		return nil
	}
	return sh.attach(s, "session-"+strconv.FormatUint(sharedExpertRingAgent.Add(1), 10), true)
}

// closeSharedExpertRings frees every model-scoped ring once, after the last session detached.
func (m *Model) closeSharedExpertRings() {
	if m == nil {
		return
	}
	sharedExpertRingsMu.Lock()
	rings := m.sharedExpertRings
	m.sharedExpertRings = nil
	sharedExpertRingsMu.Unlock()
	for _, sh := range rings {
		_ = sh.Close()
	}
}
