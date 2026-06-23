package polymodel

import (
	"errors"
	"sort"
)

// ---------------------------------------------------------------------------
// Residency — the "host many models" half.
// ---------------------------------------------------------------------------

// ModelID names a hosted model within one kernel process.
type ModelID string

// Model is the residency descriptor of one hosted model.
//
// WeightBytes is the resident HBM footprint AND the proxy for the model's
// per-token decode cost: a decode step streams the whole weight set from memory
// once per token, so the scarce, serial resource is weight bandwidth. Family
// groups models that share a tokenizer + embedding (and therefore a valid
// draft-token vocabulary) — it is the eligibility key for ensemble speculation;
// "" means unique (no peer may draft for it). Pinned models are exempt from LRU
// eviction (e.g. an always-warm shared drafter/verifier).
type Model struct {
	ID          ModelID
	Family      string
	WeightBytes int64
	Pinned      bool
}

// Admission errors. Both error paths leave the pool UNCHANGED.
var (
	ErrEmptyID      = errors.New("polymodel: model has empty ID")
	ErrTooLarge     = errors.New("polymodel: model alone exceeds the pool budget")
	ErrPinnedNoRoom = errors.New("polymodel: cannot fit model; remaining residents are pinned")
)

type entry struct {
	m    Model
	used uint64 // logical-clock recency stamp; higher == more recently used
}

// Pool holds many prefill-warm models under one weight-byte budget. The budget,
// not an architectural cap, bounds how many models stay warm; admitting past it
// evicts the coldest UNPINNED model (LRU) so a hot working set survives. The pool
// tracks only residency + recency — it owns no weights and no KV (the model leaf
// does); it is the bookkeeping that lets a kernel keep 10s of models warm and
// reason about which to drop.
type Pool struct {
	budget int64
	used   int64
	clock  uint64
	models map[ModelID]*entry
}

// NewPool returns an empty pool with the given weight-byte budget.
func NewPool(budgetBytes int64) *Pool {
	if budgetBytes < 0 {
		budgetBytes = 0
	}
	return &Pool{budget: budgetBytes, models: map[ModelID]*entry{}}
}

func (p *Pool) tick() uint64 { p.clock++; return p.clock }

// Admit makes m resident, evicting the coldest UNPINNED models (LRU) as needed to
// stay within budget, and returns the evicted IDs in eviction order. A model that
// alone exceeds the budget returns ErrTooLarge; a model that would fit only if a
// pinned resident were dropped returns ErrPinnedNoRoom. Admission is
// all-or-nothing: an error leaves the pool byte-for-byte unchanged. Re-admitting
// an already-resident model is a Touch (the descriptor is immutable — Evict then
// Admit to change WeightBytes/Pinned/Family).
func (p *Pool) Admit(m Model) ([]ModelID, error) {
	if m.ID == "" {
		return nil, ErrEmptyID
	}
	if m.WeightBytes < 0 {
		m.WeightBytes = 0
	}
	if e, ok := p.models[m.ID]; ok {
		e.used = p.tick()
		return nil, nil
	}
	if m.WeightBytes > p.budget {
		return nil, ErrTooLarge
	}
	// Feasibility check WITHOUT mutating: can the unpinned residents free enough?
	if need := p.used + m.WeightBytes - p.budget; need > 0 {
		var evictable int64
		for _, e := range p.models {
			if !e.m.Pinned {
				evictable += e.m.WeightBytes
			}
		}
		if evictable < need {
			return nil, ErrPinnedNoRoom
		}
	}
	// Evict coldest-unpinned-first until it fits (feasibility guaranteed a path).
	var evicted []ModelID
	for p.used+m.WeightBytes > p.budget {
		victim := p.coldestUnpinned()
		evicted = append(evicted, victim)
		p.used -= p.models[victim].m.WeightBytes
		delete(p.models, victim)
	}
	p.models[m.ID] = &entry{m: m, used: p.tick()}
	p.used += m.WeightBytes
	return evicted, nil
}

// coldestUnpinned returns the least-recently-used unpinned model, tie-broken by
// ID so the choice is deterministic regardless of map iteration order. Returns ""
// only when every resident is pinned (Admit's feasibility check rules that out
// before this is reached on a fitting path).
func (p *Pool) coldestUnpinned() ModelID {
	best := ModelID("")
	var bestUsed uint64
	for id, e := range p.models {
		if e.m.Pinned {
			continue
		}
		if best == "" || e.used < bestUsed || (e.used == bestUsed && id < best) {
			best, bestUsed = id, e.used
		}
	}
	return best
}

// Touch marks a model as most-recently-used (call on a prefill or decode). It is
// the LRU signal that keeps a hot model warm. Returns false if not resident.
func (p *Pool) Touch(id ModelID) bool {
	e, ok := p.models[id]
	if ok {
		e.used = p.tick()
	}
	return ok
}

// Evict removes a model and frees its bytes. Returns false if not resident.
func (p *Pool) Evict(id ModelID) bool {
	e, ok := p.models[id]
	if !ok {
		return false
	}
	p.used -= e.m.WeightBytes
	delete(p.models, id)
	return true
}

// Has reports whether a model is resident.
func (p *Pool) Has(id ModelID) bool { _, ok := p.models[id]; return ok }

// Get returns a resident model's descriptor.
func (p *Pool) Get(id ModelID) (Model, bool) {
	e, ok := p.models[id]
	if !ok {
		return Model{}, false
	}
	return e.m, true
}

// Used returns the resident weight bytes; it is always <= Budget (the invariant
// the witness suite asserts across an arbitrary admit sequence).
func (p *Pool) Used() int64 { return p.used }

// Budget returns the configured weight-byte budget.
func (p *Pool) Budget() int64 { return p.budget }

// Len returns the number of resident models.
func (p *Pool) Len() int { return len(p.models) }

// Resident returns the resident model IDs sorted by ID (deterministic).
func (p *Pool) Resident() []ModelID {
	out := make([]ModelID, 0, len(p.models))
	for id := range p.models {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// ---------------------------------------------------------------------------
// The decode lane — the "decode one" half.
// ---------------------------------------------------------------------------

// Phase distinguishes the two cost regimes the whole design turns on: Prefill is
// compute-bound and interleavable; Decode is HBM-bandwidth-bound and serial.
type Phase int

const (
	// Prefill is the compute-bound, parallelizable, shareable half.
	Prefill Phase = iota
	// Decode is the bandwidth-bound half that must run one model at a time.
	Decode
)

// String renders a Phase for test output and traces.
func (p Phase) String() string {
	if p == Prefill {
		return "prefill"
	}
	return "decode"
}

// Request is one unit of work for a hosted model: a prompt to prefill and a
// number of tokens to decode. Priority orders the decode lane; Seq is the arrival
// order used as an FCFS tie-break (a logical clock, never wall time — so a
// schedule is reproducible).
type Request struct {
	Model    ModelID
	Prefill  int
	Decode   int
	Priority int
	Seq      uint64
}

// Step is one unit of a schedule: a model running a phase for some tokens.
type Step struct {
	Model  ModelID
	Phase  Phase
	Tokens int
}

// betterDecoder reports whether a should own the decode lane before b: higher
// Priority, then lower Seq (FCFS), then lower ModelID — total and deterministic.
func betterDecoder(a, b Request) bool {
	if a.Priority != b.Priority {
		return a.Priority > b.Priority
	}
	if a.Seq != b.Seq {
		return a.Seq < b.Seq
	}
	return a.Model < b.Model
}

// NextDecoder picks which RESIDENT model owns the single decode lane next, by the
// betterDecoder order. A request whose model is not in pool is ineligible (you
// cannot decode weights you have evicted); pass nil pool to skip the residency
// filter. Returns the index into reqs of the chosen request, or -1 if none has
// decode work left and is eligible.
func NextDecoder(reqs []Request, pool *Pool) int {
	best := -1
	for i := range reqs {
		r := reqs[i]
		if r.Decode <= 0 {
			continue
		}
		if pool != nil && !pool.Has(r.Model) {
			continue
		}
		if best < 0 || betterDecoder(r, reqs[best]) {
			best = i
		}
	}
	return best
}

// Stats summarizes a schedule. MaxConcurrentDecode is the load-bearing invariant:
// it is always 1 (the lane is serial), guaranteed by construction in Schedule and
// asserted by the witness suite.
type Stats struct {
	PrefillTokens       int
	DecodeTokens        int
	DecodeSteps         int
	MaxConcurrentDecode int
}

// Schedule turns a set of requests into a deterministic execution plan that
// honors the asymmetry: every request's prefill is emitted once (compute-bound,
// independent), then the decode work is round-robined across requests in
// betterDecoder order, at most `quantum` decode tokens per turn, until every
// request's decode budget is drained. Decode steps are strictly serial — one
// model per step — so the plan is a faithful single-lane decode schedule. It is
// pure (no pool, no clock): it assumes each request's model is resident, which the
// caller arranges via Pool. quantum <= 0 defaults to 1.
func Schedule(reqs []Request, quantum int) ([]Step, Stats) {
	if quantum <= 0 {
		quantum = 1
	}
	order := make([]int, len(reqs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return betterDecoder(reqs[order[i]], reqs[order[j]])
	})

	steps := make([]Step, 0, len(reqs))
	var st Stats
	remaining := make([]int, len(reqs))
	// Prefill emission first, in decode order so the plan reads top-priority-first.
	for _, idx := range order {
		r := reqs[idx]
		if r.Prefill > 0 {
			steps = append(steps, Step{Model: r.Model, Phase: Prefill, Tokens: r.Prefill})
			st.PrefillTokens += r.Prefill
		}
		remaining[idx] = r.Decode
		if r.Decode > 0 {
			st.DecodeTokens += r.Decode
		}
	}
	// Round-robin the serial decode lane: one model per step, quantum tokens each.
	for {
		progressed := false
		for _, idx := range order {
			if remaining[idx] <= 0 {
				continue
			}
			n := quantum
			if remaining[idx] < n {
				n = remaining[idx]
			}
			steps = append(steps, Step{Model: reqs[idx].Model, Phase: Decode, Tokens: n})
			remaining[idx] -= n
			st.DecodeSteps++
			progressed = true
		}
		if !progressed {
			break
		}
	}
	if st.DecodeSteps > 0 {
		st.MaxConcurrentDecode = 1
	}
	return steps, st
}

// DecodeBandwidthBytes is the HBM traffic a plan's decode steps cost: each decoded
// token streams its model's whole weight set once, so the cost is the sum over
// Decode steps of (tokens × WeightBytes). It quantifies WHY decode must serialize
// — N models decoding concurrently would multiply this against a fixed memory bus
// — and why holding many warm is cheap: residency is capacity, only the decoding
// model pays bandwidth. A model missing from weights contributes 0.
func DecodeBandwidthBytes(steps []Step, weights map[ModelID]int64) int64 {
	var total int64
	for _, s := range steps {
		if s.Phase != Decode {
			continue
		}
		total += int64(s.Tokens) * weights[s.Model]
	}
	return total
}

// ---------------------------------------------------------------------------
// Cache-led multi-token prediction — the "next-gen MTP" half.
// ---------------------------------------------------------------------------

// SpecResult is the outcome of verifying a K-token draft against a target model in
// ONE prefill-shaped verify pass (the target processes all K proposed tokens in
// parallel — compute-bound — instead of K serial bandwidth-bound decode steps).
// The KEEP/EVICT counts map directly onto the model leaf's bit-exact KV primitives.
type SpecResult struct {
	Accepted int // leading draft tokens that matched the target's argmax
	Advance  int // REAL tokens committed this round (Accepted + 1 correction/bonus)
	KeepKV   int // speculative KV positions to KEEP  (== Accepted) → cache survivors
	EvictKV  int // speculative KV positions to ROLL BACK (== K-Accepted) → KVCache.Evict
}

// AcceptGreedy implements the greedy (argmax) speculative-decoding accept rule.
// Given a draft of proposed token ids and the target model's argmax token at each
// of those positions (from the single verify pass), it accepts the longest
// matching prefix; the first divergence is replaced by the target's own token (the
// correction), so a fully-rejected draft still advances by 1 REAL token, and a
// fully-accepted draft of K advances by K+1 when the verify pass carried a bonus
// position (targetArgmax longer than draft).
//
// The KV accounting is the cache-led half: KeepKV (== Accepted) speculative
// positions stay; EvictKV (== K-Accepted) drafted-but-rejected positions are
// rolled back with model.KVCache.Evict — bit-exact to never having drafted them,
// the rollback page-shared engines cannot do exactly. This is the proven core; the
// GPU verify pass and the Evict call are the engine wiring (see the plan doc).
func AcceptGreedy(draft, targetArgmax []int) SpecResult {
	k := len(draft)
	limit := k
	if len(targetArgmax) < limit {
		limit = len(targetArgmax)
	}
	accepted := 0
	for accepted < limit && draft[accepted] == targetArgmax[accepted] {
		accepted++
	}
	res := SpecResult{Accepted: accepted, KeepKV: accepted, EvictKV: k - accepted}
	// Advance = accepted real tokens + 1 correction/bonus, present whenever the
	// verify pass produced a token at the divergence (or bonus) position.
	if accepted < len(targetArgmax) {
		res.Advance = accepted + 1
	} else {
		res.Advance = accepted
	}
	return res
}

// PickDrafter chooses, among the OTHER resident models, the best speculator for the
// active decoder — the "idle co-resident models become the speculation ensemble"
// idea. Eligibility: a DIFFERENT model in the SAME non-empty Family (shared
// tokenizer, so its token ids are valid drafts for the target). Among eligible it
// prefers the cheapest (smallest WeightBytes, so drafting is fast), tie-broken by
// ID. Returns "" when the active model is not resident, has no Family, or no
// eligible drafter is warm — in which case the target self-decodes (no speculation,
// always correct).
func PickDrafter(active ModelID, pool *Pool) ModelID {
	if pool == nil {
		return ""
	}
	want, ok := pool.Get(active)
	if !ok || want.Family == "" {
		return ""
	}
	best := ModelID("")
	var bestBytes int64
	for _, id := range pool.Resident() { // sorted → deterministic
		if id == active {
			continue
		}
		m, _ := pool.Get(id)
		if m.Family != want.Family {
			continue
		}
		if best == "" || m.WeightBytes < bestBytes {
			best, bestBytes = id, m.WeightBytes
		}
	}
	return best
}

// EffectiveTokensPerVerify is the expected number of REAL tokens a single target
// verify pass advances, given a draft length k and a per-token acceptance
// probability a in [0,1], under the greedy accept rule:
//
//	E = sum_{i=0}^{k} a^i = (1 - a^(k+1)) / (1 - a)   (a < 1);   E = k+1   (a == 1)
//
// This is the classic speculative-decoding speedup model (Leviathan et al.): one
// bandwidth-bound verify yields E real tokens instead of 1, so on the single
// serial decode lane multi-token prediction multiplies throughput by ~E (before
// subtracting draft cost). It is the honest arithmetic behind any speedup claim —
// no measurement. k <= 0 returns 1 (a plain decode); a is clamped to [0,1].
func EffectiveTokensPerVerify(k int, a float64) float64 {
	if k <= 0 {
		return 1
	}
	switch {
	case a < 0:
		a = 0
	case a > 1:
		a = 1
	}
	if a == 1 {
		return float64(k + 1)
	}
	pow := 1.0
	for i := 0; i < k+1; i++ {
		pow *= a
	}
	return (1 - pow) / (1 - a)
}
