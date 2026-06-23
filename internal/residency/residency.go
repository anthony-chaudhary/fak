package residency

import (
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// ErrNilWeights is returned by Admit when the supplied *model.Model is nil — a
// resident entry must bind a real weight handle, which is the whole point of this
// layer: the descriptor→weights binding that polymodel.Pool deliberately does not own.
var ErrNilWeights = errors.New("residency: admit requires a non-nil *model.Model")

// Evicted is one resident paged out by an Admit or Evict: its ID and the weight handle
// the caller should now release / move off the resident device. polymodel.Pool returns
// only the evicted IDs (it owns no weights); this layer owns the binding, so it hands
// the evicted *model.Model back — the page-out signal.
type Evicted struct {
	ID      polymodel.ModelID
	Weights *model.Model
}

// Manager hosts many prefill-warm *model.Model under one resident weight-byte budget,
// reusing polymodel.Pool as the budget + LRU-eviction POLICY and binding each admitted
// residency descriptor to the real in-kernel weights it governs. polymodel.Pool owns
// only residency + recency bookkeeping ("it owns no weights and no KV (the model leaf
// does)"); Manager is the layer that lifts the single-*model.Model assumption and holds
// the weight handles, handing them back on eviction so a caller can page them out / free
// their resident memory. It is the rung-#531 half of "host many models on one backend":
// the policy is proven (polymodel), the weight binding + page-out hand-back is new.
//
// The budget, the LRU victim choice, the pinned-exemption, and the all-or-nothing admit
// semantics are ALL polymodel.Pool's — Manager does not re-implement any of them; it
// delegates and binds. So every invariant the polymodel witness suite asserts
// (used<=budget, pinned-never-evicted, admit-unchanged-on-error) holds here by
// construction. A Manager is safe for concurrent use (serve is multi-goroutine).
type Manager struct {
	mu      sync.Mutex
	pool    *polymodel.Pool
	weights map[polymodel.ModelID]*model.Model
}

// New returns an empty Manager with the given resident weight-byte budget. A negative
// budget is clamped to 0 (matching polymodel.NewPool), so every Admit pages out.
func New(budgetBytes int64) *Manager {
	return &Manager{pool: polymodel.NewPool(budgetBytes), weights: map[polymodel.ModelID]*model.Model{}}
}

// Admit makes m resident under id, binding the residency descriptor (id, weightBytes,
// family, prefixDigest, pinned — the cross-model prefill-share and speculation keys) to
// the real weight handle m. It evicts the coldest UNPINNED residents (LRU) as needed to
// stay within budget and returns those evicted (weight handles handed back for page-out)
// in eviction order. The budget test and the LRU victim selection are polymodel.Pool's;
// this method only binds the descriptor to m and releases the evicted handles. On any
// error the resident set is byte-for-byte unchanged (polymodel.Pool's all-or-nothing
// admit): a nil m returns ErrNilWeights; a model larger than the budget returns
// polymodel.ErrTooLarge; a model that fits only by dropping a pinned resident returns
// polymodel.ErrPinnedNoRoom. Re-admitting an already-resident id is a Touch (the
// descriptor is immutable — Evict then Admit to change it).
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

// Get returns the resident weight handle for id (a pure lookup; it does not update
// recency). Call Touch on a decode so a hot model stays warm — exactly mirroring the
// polymodel.Pool.Get / Touch split. Returns (nil, false) for a non-resident id.
func (r *Manager) Get(id polymodel.ModelID) (*model.Model, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.weights[id]
	return m, ok
}

// Touch marks id most-recently-used — the LRU signal that keeps a hot model warm.
// Returns false if id is not resident.
func (r *Manager) Touch(id polymodel.ModelID) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Touch(id)
}

// Evict explicitly removes id, handing its weight handle back for the caller to page
// out / free. Returns (nil, false) if id was not resident.
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

// Descriptor returns the residency descriptor (Family / WeightBytes / Pinned /
// PrefixDigest) for a resident model — the polymodel.Pool.Get view. Returns
// (zero, false) for a non-resident id.
func (r *Manager) Descriptor(id polymodel.ModelID) (polymodel.Model, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Get(id)
}

// Resident returns the resident model IDs sorted by ID (deterministic).
func (r *Manager) Resident() []polymodel.ModelID {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pool.Resident()
}

// Len returns the number of resident models.
func (r *Manager) Len() int { r.mu.Lock(); defer r.mu.Unlock(); return r.pool.Len() }

// Used returns the resident weight bytes; always <= Budget (the polymodel invariant).
func (r *Manager) Used() int64 { r.mu.Lock(); defer r.mu.Unlock(); return r.pool.Used() }

// Budget returns the configured resident weight-byte budget.
func (r *Manager) Budget() int64 { r.mu.Lock(); defer r.mu.Unlock(); return r.pool.Budget() }
