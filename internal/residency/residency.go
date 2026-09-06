package residency

import (
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

var (
	// ErrNilWeights is returned by Admit when the supplied *model.Model is nil.
	ErrNilWeights = errors.New("residency: admit requires a non-nil *model.Model")

	// ErrModelAlreadyResident is returned by Admit when attempting to re-admit an
	// existing model with a different model handle without first evicting.
	ErrModelAlreadyResident = errors.New("residency: model already resident with different weights")

	// ErrDescriptorMismatch is returned by Admit when attempting to re-admit an
	// existing model with a mismatched descriptor without first evicting.
	ErrDescriptorMismatch error = descriptorMismatchErr{}
)

type descriptorMismatchErr struct{}

func (descriptorMismatchErr) Error() string {
	return "residency: descriptor mismatch on re-admission"
}

func (descriptorMismatchErr) Unwrap() error {
	return ErrModelAlreadyResident
}

func (descriptorMismatchErr) Is(target error) bool {
	return target == ErrDescriptorMismatch || target == ErrModelAlreadyResident
}

// Evicted represents a resident model paged out by Admit, Evict, or SetBudget.
type Evicted struct {
	ID      polymodel.ModelID
	Weights *model.Model
}

// Manager hosts resident *model.Model weights under a configured byte budget
// with LRU eviction, delegating budget and eviction policy to polymodel.Pool.
// All methods are safe for concurrent use.
type Manager struct {
	mu      sync.Mutex
	pool    *polymodel.Pool
	weights map[polymodel.ModelID]*model.Model
}

// New returns a Manager with the given resident weight-byte budget.
// Negative budgets are clamped to 0.
func New(budgetBytes int64) *Manager {
	return &Manager{
		pool:    polymodel.NewPool(budgetBytes),
		weights: make(map[polymodel.ModelID]*model.Model),
	}
}

// Admit places model m under id into the resident set, evicting the coldest
// unpinned models as needed to remain within the configured budget.
//
// Returns ErrNilWeights if m is nil. If admission fails, the resident set
// remains unchanged. Re-admitting an existing ID with identical weights and
// descriptor updates its LRU recency (Touch). Re-admitting an existing ID
// with a different model handle or mismatched descriptor returns an error
// (ErrModelAlreadyResident or ErrDescriptorMismatch) without mutating state.
func (r *Manager) Admit(id polymodel.ModelID, m *model.Model, weightBytes int64, family, prefixDigest string, pinned bool) ([]Evicted, error) {
	if m == nil {
		return nil, ErrNilWeights
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	desc := polymodel.Model{
		ID:           id,
		Family:       family,
		WeightBytes:  weightBytes,
		Pinned:       pinned,
		PrefixDigest: prefixDigest,
	}
	if desc.WeightBytes < 0 {
		desc.WeightBytes = 0
	}
	if existingM, ok := r.weights[id]; ok {
		existingDesc, _ := r.pool.Get(id)
		if existingDesc != desc {
			return nil, ErrDescriptorMismatch
		}
		if existingM != m {
			return nil, ErrModelAlreadyResident
		}
	}
	evictedIDs, err := r.pool.Admit(desc)
	if err != nil {
		return nil, err
	}
	r.weights[id] = m
	out := make([]Evicted, 0, len(evictedIDs))
	for _, vid := range evictedIDs {
		out = append(out, Evicted{ID: vid, Weights: r.weights[vid]})
		delete(r.weights, vid)
	}
	return out, nil
}

// Get returns the resident weight handle for id without updating recency.
func (r *Manager) Get(id polymodel.ModelID) (*model.Model, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.weights[id]
	return m, ok
}

// Touch marks id as most recently used. Returns false if id is not resident.
func (r *Manager) Touch(id polymodel.ModelID) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Touch(id)
}

// Evict removes id from the resident pool and returns its weight handle.
// Returns false if id is not resident.
func (r *Manager) Evict(id polymodel.ModelID) (*model.Model, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.pool.Evict(id) {
		return nil, false
	}
	m := r.weights[id]
	delete(r.weights, id)
	return m, true
}

// SetBudget adjusts the resident weight-byte budget at runtime.
// If shrinking requires evictions, coldest unpinned models are returned.
// If pinned models exceed the new budget, ErrPinnedNoRoom is returned and
// the resident set is unchanged.
func (r *Manager) SetBudget(newBudgetBytes int64) ([]Evicted, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	evictedIDs, err := r.pool.Resize(newBudgetBytes)
	if err != nil {
		return nil, err
	}
	out := make([]Evicted, 0, len(evictedIDs))
	for _, vid := range evictedIDs {
		out = append(out, Evicted{ID: vid, Weights: r.weights[vid]})
		delete(r.weights, vid)
	}
	return out, nil
}

// Descriptor returns the residency descriptor for id.
func (r *Manager) Descriptor(id polymodel.ModelID) (polymodel.Model, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Get(id)
}

// Resident returns all resident model IDs in deterministic sorted order.
func (r *Manager) Resident() []polymodel.ModelID {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Resident()
}

// Len returns the number of resident models.
func (r *Manager) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Len()
}

// Used returns total resident weight bytes.
func (r *Manager) Used() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Used()
}

// Budget returns the configured resident weight-byte budget.
func (r *Manager) Budget() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Budget()
}
