// Package resourcelifecycle provides one typed lifecycle for model and agent resources.
//
// Invariant: resource lifecycle transitions are fail-closed and deterministic across all state transitions.
// Guard: incomplete claims missing identity, compatibility, or isolation bounds are unconditionally rejected.
package resourcelifecycle

import (
	"errors"
	"fmt"
	"sync"
)

// Ref uniquely identifies an allocated or tracked resource reference within the manager.
type Ref uint64

// Geometry specifies spatial dimensions, memory alignment, and page size requirements.
type Geometry struct {
	Shape                []int
	Alignment, PageBytes int64
}

// Claim declares resource attributes, isolation boundaries, lifetimes, and dependencies.
type Claim struct {
	Kind, Owner, Isolation, Lifetime, Compatibility, Sensitivity string
	Mutable, Shareable                                           bool
	Bytes                                                        int64
	Geometry                                                     Geometry
	AllowedLocality                                              []string
	Dependencies                                                 []Ref
	Invalidates                                                  []string
}

// Allocation associates an allocated reference with its original claim and resolved localities.
type Allocation struct {
	Ref                             Ref
	Claim                           Claim
	PlannedLocality, ActualLocality string
}

// Observation records an action event, byte accounting, and lineage delta against a managed reference.
type Observation struct {
	Ref                                        Ref
	Action                                     string
	ActualBytes, TransferBytes, RecomputeBytes int64
	From, To, Reason                           string
}

// Receipt exposes current audit state and aggregated transfer metrics for a tracked resource.
type Receipt struct {
	key                                          string
	Ref                                          Ref
	Kind, Owner, PlannedLocality, ActualLocality string
	TransferBytes, RecomputeBytes                int64
	Reused, Released                             bool
	Observations                                 []Observation
}

// Manager coordinates concurrency-safe allocation, caching, reuse, and disposal of resources.
// Invariant: all mutable mutations hold the manager write lock; reads hold the read lock.
type Manager struct {
	mu    sync.RWMutex
	next  Ref
	byRef map[Ref]*Receipt
	byKey map[string]Ref
}

// ErrIsolation signals an incompatible isolation boundary across resource consumers.
var ErrIsolation = errors.New("resourcelifecycle: incompatible isolation")

// New instantiates an initialized resource manager ready to resolve allocations.
func New() *Manager      { return &Manager{byRef: map[Ref]*Receipt{}, byKey: map[string]Ref{}} }
func key(c Claim) string { return c.Kind + "|" + c.Compatibility + "|" + c.Isolation }

// Resolve admits or reuses a claim matching isolation and compatibility constraints.
// Guard: requires complete non-empty kind, owner, compatibility, and isolation attributes.
func (m *Manager) Resolve(c Claim, planned, actual string) (Allocation, error) {
	if c.Kind == "" || c.Owner == "" || c.Compatibility == "" || c.Isolation == "" {
		return Allocation{}, errors.New("resourcelifecycle: incomplete claim")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if ref := m.byKey[key(c)]; ref != 0 && c.Shareable && !c.Mutable {
		r := m.byRef[ref]
		r.Reused = true
		r.Observations = append(r.Observations, Observation{Ref: ref, Action: "reuse", ActualBytes: r.Observations[0].ActualBytes})
		return Allocation{Ref: ref, Claim: c, PlannedLocality: planned, ActualLocality: r.ActualLocality}, nil
	}
	m.next++
	a := Allocation{Ref: m.next, Claim: c, PlannedLocality: planned, ActualLocality: actual}
	r := &Receipt{key: key(c), Ref: a.Ref, Kind: c.Kind, Owner: c.Owner, PlannedLocality: planned, ActualLocality: actual, Observations: []Observation{{Ref: a.Ref, Action: "allocate", ActualBytes: c.Bytes, To: actual}}}
	if planned != actual {
		r.TransferBytes = c.Bytes
		r.Observations = append(r.Observations, Observation{Ref: a.Ref, Action: "transfer", ActualBytes: c.Bytes, TransferBytes: c.Bytes, From: planned, To: actual})
	}
	m.byRef[a.Ref] = r
	m.byKey[key(c)] = a.Ref
	return a, nil
}

// Observe records an operational event (such as transfer, recompute, or eviction) against an existing ref.
func (m *Manager) Observe(o Observation) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	r := m.byRef[o.Ref]
	if r == nil {
		return fmt.Errorf("unknown ref %d", o.Ref)
	}
	r.Observations = append(r.Observations, o)
	r.TransferBytes += o.TransferBytes
	r.RecomputeBytes += o.RecomputeBytes
	if o.Action == "release" || o.Action == "evict" {
		r.Released = true
		delete(m.byKey, r.key)
	}
	return nil
}

// Get retrieves a defensive copy of the current audit receipt for a registered resource ref.
func (m *Manager) Get(ref Ref) (Receipt, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	r, ok := m.byRef[ref]
	if !ok {
		return Receipt{}, false
	}
	out := *r
	out.Observations = append([]Observation(nil), r.Observations...)
	return out, true
}

// Teardown releases all non-released resources belonging to the specified owner.
func (m *Manager) Teardown(owner string) []Receipt {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Receipt
	for _, r := range m.byRef {
		if r.Owner == owner && !r.Released {
			r.Released = true
			r.Observations = append(r.Observations, Observation{Ref: r.Ref, Action: "release", Reason: "owner_teardown"})
			out = append(out, *r)
		}
	}
	return out
}
