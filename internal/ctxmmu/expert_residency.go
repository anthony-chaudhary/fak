package ctxmmu

import (
	"errors"
	"sort"
)

// expert_residency.go — the MoE expert-residency policy for the unified-memory serving tier
// (anthony-chaudhary/fak#12952, parent #12640).
//
// Routed experts are ~56.6% of the DeepSeek-V4.1 checkpoint, so whether a single Halo decode
// is disk-bound is decided by residency policy rather than by kernel throughput. Two gaps
// made the shipped residency path unable to express that policy:
//
//  1. There was no way to keep a PRE-FILL sweep from evicting the resident decode hot set.
//     A prefill touches a broad, one-shot expert union; under a single cache those cold
//     touches evict the genuinely hot decode experts, so every request pays a cold start.
//  2. There was no trace-ranked warm start: the resident set was never seeded from what a
//     prior workload actually routed. Coding-vs-general top-25% overlap is only 0.18–0.31
//     Jaccard (upstream-quoted, unverified here), so the workload profile is a real lever.
//
// This file adds the policy seam the #12640 "bounded expert/Engram streaming and cache
// accounting" bullet names: a resident hot-set LRU plus a SEPARATE bounded transient ring
// used for prefill sweeps, transient→resident promotion by DESCRIPTOR SWAP (no payload
// re-read), and a trace-ranked warm start that seeds the resident set behind the trace.
//
// Scope (honest, matching the sibling residency layers' own): a standalone policy +
// accounting primitive, OFF the live serve path. It moves no bytes and links nothing new
// into the forward. The payload-re-read claim is modelled by the ResidencyBacking interface,
// so it is provable here with a counting fake and remains reproducible when a real
// disk-backed store is wired behind the same interface. A real-workload warm-start benefit
// is [HW-WITNESSED] and explicitly out of scope (see the issue's gold-plating boundary).

// ErrExpertResidencyBudget is returned when a ring's byte budget cannot hold a single entry.
var ErrExpertResidencyBudget = errors.New("ctxmmu: expert residency budget too small for one descriptor")

// ResidencyTier names which slot of the policy an expert currently occupies.
type ResidencyTier string

const (
	// TierCold is an expert held by neither ring — a miss re-reads it from backing.
	TierCold ResidencyTier = "cold"
	// TierTransient is an expert resident only in the transient (prefill) ring.
	TierTransient ResidencyTier = "transient"
	// TierResident is an expert resident in the hot-set (decode) ring.
	TierResident ResidencyTier = "resident"
)

// ExpertDescriptor is the identity + size + opaque payload handle for one resident expert
// weight. Payload is the reusable payload the backing store would otherwise have to
// re-read; promotion swaps the descriptor carrying it rather than reloading bytes.
type ExpertDescriptor struct {
	Layer   int
	Expert  int
	Bytes   int64
	Payload any
}

// ExpertResidencyOptions configures a policy. A non-positive HotBytes or TransientBytes is
// replaced by the corresponding default; the hot budget must admit at least one descriptor.
type ExpertResidencyOptions struct {
	// HotBytes is the byte budget of the resident hot-set (decode) ring.
	HotBytes int64
	// TransientBytes is the byte budget of the separate prefill transient ring. It is NOT
	// carved from HotBytes: a prefill sweep can only churn this ring, never the hot set.
	TransientBytes int64
}

// Default budgets for a 128 GB Halo-class appliance: a 6 GiB resident hot set and a 2 GiB
// prefill transient ring. They are placeholders for the operator's measured sizing
// decision (study rows X7/X8/X9), not a claim; a caller may override either.
const (
	DefaultExpertHotBytes       int64 = 6 << 30
	DefaultExpertTransientBytes int64 = 2 << 30
)

// ExpertResidencyStats is the accounting surface the private serving consumer reports. A
// transient ring hit is tracked separately from a resident hit so a prefill sweep's churn is
// measurable rather than indistinguishable from useful decode reuse.
type ExpertResidencyStats struct {
	ResidentHits     int64 `json:"resident_hits"`
	TransientHits    int64 `json:"transient_hits"`
	ColdMisses       int64 `json:"cold_misses"`
	Promotions       int64 `json:"promotions"`
	HotEvictions     int64 `json:"hot_evictions"`
	TransientEvicts  int64 `json:"transient_evictions"`
	ResidentBytes    int64 `json:"resident_bytes"`
	TransientBytes   int64 `json:"transient_bytes"`
	ResidentEntries  int   `json:"resident_entries"`
	TransientEntries int   `json:"transient_entries"`
}

// ResidencyBacking is the slow store a cold miss must read. Loader is consulted ONLY on a
// cold miss (or to seed warm start), so a resident/transient hit proves no backing re-read.
type ResidencyBacking interface {
	// Load returns the descriptor for (layer, expert). found=false is a cold-miss refusal.
	Load(layer, expert int) (ExpertDescriptor, bool)
}

// ExpertResidencyPolicy holds a resident hot-set LRU and a separate bounded transient ring.
// A prefill sweep touches the transient ring; a decode touch promotes into the hot set.
type ExpertResidencyPolicy struct {
	opts      ExpertResidencyOptions
	resident  *expertLRURing
	transient *expertLRURing
	backing   ResidencyBacking
	stats     ExpertResidencyStats
}

type expertResidencyKey struct{ layer, expert int }

// NewExpertResidencyPolicy builds a policy over backing. A nil backing is admitted: a
// synthetic warm-start caller may provide descriptors through Load anyway.
func NewExpertResidencyPolicy(opts ExpertResidencyOptions, backing ResidencyBacking) (*ExpertResidencyPolicy, error) {
	if opts.HotBytes <= 0 {
		opts.HotBytes = DefaultExpertHotBytes
	}
	if opts.TransientBytes <= 0 {
		opts.TransientBytes = DefaultExpertTransientBytes
	}
	return &ExpertResidencyPolicy{
		opts:      opts,
		resident:  newExpertLRURing(opts.HotBytes),
		transient: newExpertLRURing(opts.TransientBytes),
		backing:   backing,
	}, nil
}

// Prefill touches an expert during a prompt sweep. It consults the hot set first (a sweep
// that hits a resident expert must not disturb it), then a resident transient entry (LRU
// touch, no backing read), otherwise stages a cold descriptor into the TRANSIENT ring. It
// never inserts into the hot set, so a prefill cannot evict the decode hot set; a prefill
// cannot promote. Returns the descriptor when a copy is resident/loaded, else ok=false.
func (p *ExpertResidencyPolicy) Prefill(layer, expert int) (ExpertDescriptor, bool) {
	if p == nil {
		return ExpertDescriptor{}, false
	}
	key := expertResidencyKey{layer, expert}
	if d, ok := p.resident.get(key); ok {
		p.stats.ResidentHits++
		return d, true
	}
	if d, ok := p.transient.touch(key); ok {
		p.stats.TransientHits++
		return d, true
	}
	d, ok := p.load(layer, expert)
	if !ok {
		p.stats.ColdMisses++
		return ExpertDescriptor{}, false
	}
	p.stageTransient(key, d)
	return d, true
}

// Decode touches an expert during post-prefill token generation. A hot-set hit is decoded
// directly. A transient hit PROMOTES the descriptor into the hot set by overlap: swapping
// the same descriptor in, with no backing read. A cold touch loads into the hot set.
func (p *ExpertResidencyPolicy) Decode(layer, expert int) (ExpertDescriptor, bool) {
	if p == nil {
		return ExpertDescriptor{}, false
	}
	key := expertResidencyKey{layer, expert}
	if d, ok := p.resident.touch(key); ok {
		p.stats.ResidentHits++
		return d, true
	}
	if d, ok := p.transient.remove(key); ok {
		p.stats.TransientHits++
		p.promote(key, d)
		p.stats.Promotions++
		return d, true
	}
	d, ok := p.load(layer, expert)
	if !ok {
		p.stats.ColdMisses++
		return ExpertDescriptor{}, false
	}
	p.promote(key, d)
	return d, true
}

// ExpertAccess is one observed (layer,expert) touch a warm-start trace is built from. It is
// deliberately ctxmmu-local: this package is a low tier that must not import internal/model,
// so a caller adapts its own trace events to this shape (the model-side adapter lives in
// internal/model/v41_expert_policy.go and the private serving consumer joins the two).
type ExpertAccess struct {
	Layer       int
	Expert      int
	WeightBytes int64
}

// WarmStart seeds the resident hot set from an ordered access trace: the most FREQUENT
// (layer,expert) identities are pre-loaded (not counted as a demand hit), so the first
// decode of a hot expert is resident. Loads are read from backing once; a descriptor a
// trace entry cannot supply is skipped. It returns the number of experts seeded.
func (p *ExpertResidencyPolicy) WarmStart(accesses []ExpertAccess) int {
	if p == nil {
		return 0
	}
	counts := map[expertResidencyKey]int64{}
	for _, ev := range accesses {
		if ev.Layer < 0 || ev.Expert < 0 {
			continue
		}
		counts[expertResidencyKey{ev.Layer, ev.Expert}]++
	}
	type ranked struct {
		key   expertResidencyKey
		count int64
	}
	order := make([]ranked, 0, len(counts))
	for k, c := range counts {
		order = append(order, ranked{k, c})
	}
	// Deterministic (count desc, layer asc, expert asc): frequency first, identity tie-break.
	sort.Slice(order, func(i, j int) bool {
		if order[i].count != order[j].count {
			return order[i].count > order[j].count
		}
		if order[i].key.layer != order[j].key.layer {
			return order[i].key.layer < order[j].key.layer
		}
		return order[i].key.expert < order[j].key.expert
	})
	seeded := 0
	for _, r := range order {
		if p.resident.has(r.key) {
			continue
		}
		d, ok := p.load(r.key.layer, r.key.expert)
		if !ok {
			continue
		}
		// Seed the budget hottest that FIT. Do not let a lower-ranked cold expert evict an
		// already-seeded hot one: warm start is a one-shot seeding, not a replay, so it stops
		// at the budget rather than churning the set it is building.
		if p.resident.used+d.Bytes > p.resident.budget {
			break
		}
		p.promote(r.key, d)
		seeded++
	}
	return seeded
}

// Stats returns a snapshot of the policy accounting.
func (p *ExpertResidencyPolicy) Stats() ExpertResidencyStats {
	if p == nil {
		return ExpertResidencyStats{}
	}
	s := p.stats
	s.ResidentEntries = p.resident.len()
	s.TransientEntries = p.transient.len()
	s.ResidentBytes = p.resident.used
	s.TransientBytes = p.transient.used
	return s
}

// Tier reports where an expert currently sits, without changing recency or stats.
func (p *ExpertResidencyPolicy) Tier(layer, expert int) ResidencyTier {
	if p == nil {
		return TierCold
	}
	key := expertResidencyKey{layer, expert}
	if p.resident.has(key) {
		return TierResident
	}
	if p.transient.has(key) {
		return TierTransient
	}
	return TierCold
}

// load reads from backing, refusing a descriptor that does not identify the requested key
// (a backing that returns the wrong expert must not silently alias two experts).
func (p *ExpertResidencyPolicy) load(layer, expert int) (ExpertDescriptor, bool) {
	if p.backing == nil {
		return ExpertDescriptor{}, false
	}
	d, ok := p.backing.Load(layer, expert)
	if !ok {
		return ExpertDescriptor{}, false
	}
	if d.Layer != layer || d.Expert != expert {
		return ExpertDescriptor{}, false
	}
	return d, true
}

// promote inserts a descriptor into the hot set, evicting hot-set victims as needed.
func (p *ExpertResidencyPolicy) promote(key expertResidencyKey, d ExpertDescriptor) {
	evicted := p.resident.insert(key, d)
	for range evicted {
		p.stats.HotEvictions++
	}
}

// stageTransient inserts a descriptor into the transient ring; the ring's own LRU order
// makes the oldest transient entry the first churned by a prefill sweep.
func (p *ExpertResidencyPolicy) stageTransient(key expertResidencyKey, d ExpertDescriptor) {
	evicted := p.transient.insert(key, d)
	for range evicted {
		p.stats.TransientEvicts++
	}
}

// expertLRURing is a byte-budgeted LRU of expert descriptors. Insertion evicts the least
// recently used entries until the budget admits the newcomer; a descriptor larger than the
// whole budget is rejected (the entry is not stored) so accounting never goes negative.
type expertLRURing struct {
	budget int64
	used   int64
	items  map[expertResidencyKey]ExpertDescriptor
	order  []expertResidencyKey
}

func newExpertLRURing(budget int64) *expertLRURing {
	return &expertLRURing{budget: budget, items: map[expertResidencyKey]ExpertDescriptor{}}
}

func (r *expertLRURing) len() int { return len(r.items) }

func (r *expertLRURing) has(key expertResidencyKey) bool {
	_, ok := r.items[key]
	return ok
}

// get returns a copy without changing recency.
func (r *expertLRURing) get(key expertResidencyKey) (ExpertDescriptor, bool) {
	d, ok := r.items[key]
	return d, ok
}

// touch returns a copy and marks the entry most-recently-used.
func (r *expertLRURing) touch(key expertResidencyKey) (ExpertDescriptor, bool) {
	d, ok := r.items[key]
	if !ok {
		return ExpertDescriptor{}, false
	}
	r.bump(key)
	return d, true
}

// remove returns and drops an entry (used by promotion to move it between rings).
func (r *expertLRURing) remove(key expertResidencyKey) (ExpertDescriptor, bool) {
	d, ok := r.items[key]
	if !ok {
		return ExpertDescriptor{}, false
	}
	delete(r.items, key)
	r.used -= d.Bytes
	for i, k := range r.order {
		if k == key {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	return d, true
}

// insert adds or refreshes an entry, evicting coldest entries until it fits. Returns the
// evicted keys.
func (r *expertLRURing) insert(key expertResidencyKey, d ExpertDescriptor) []expertResidencyKey {
	if d.Bytes > r.budget || r.budget <= 0 {
		// Too big to ever fit: never store, never evict a resident for it.
		return nil
	}
	var evicted []expertResidencyKey
	if prior, ok := r.items[key]; ok {
		r.used -= prior.Bytes
		delete(r.items, key)
	}
	for r.used+d.Bytes > r.budget && len(r.order) > 0 {
		victim := r.order[0]
		v := r.items[victim]
		r.order = r.order[1:]
		delete(r.items, victim)
		r.used -= v.Bytes
		evicted = append(evicted, victim)
	}
	r.items[key] = d
	r.used += d.Bytes
	r.order = append(r.order, key)
	return evicted
}

// bump moves key to most-recently-used.
func (r *expertLRURing) bump(key expertResidencyKey) {
	for i, k := range r.order {
		if k == key {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
	r.order = append(r.order, key)
}
