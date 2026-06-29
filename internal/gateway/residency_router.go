package gateway

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Cache-aware fleet routing POLICY — the per-worker prefix-residency index + the
// power-of-two-choices scorer that lands on the ReplicaRouter skeleton's pick()
// seam (issue #41, build-on #45). The skeleton supplies the replica set and the
// dispatch path; this file supplies only the PLACEMENT POLICY: given a request's
// shared prefix and the live worker set, which worker already holds (or should
// home) that prefix's KV so the prefill is reused instead of recomputed cold.
//
// It is the SHARED substrate the issue names: one residency index, two emitters.
//   - Track A (RIDE): an external engine's KV-cache-events (vLLM KV-events / SGLang
//     RadixAttention signal) lower into KVCacheEvent and are folded by Ingest.
//   - Track B (NATIVE): fak's own internal/radixkv per-worker residency is pulled
//     in through the ResidentPrefixSource seam (IngestResidentPrefixes) without the
//     gateway importing the cache implementation.
// Both populate the SAME PrefixResidencyIndex schema; the scorer is emitter-blind.
//
// SOTA posture (not a strawman): SGLang Router keeps an APPROXIMATE per-worker radix
// tree, scores longest-prefix overlap, and falls back to plain load-balancing once
// the per-worker trees skew past a balancing threshold — so shared-prefix traffic
// does not herd onto one hot replica. This file mirrors that policy class. It is
// control-plane signal only: no KV bytes move here (that is the native KV-transport
// seed, #29), and the prefill/decode role split is its own seed (#28); this index is
// THAT seed's documented prerequisite (see ResidencyView).

// residencyOp tags a KV-cache-event: a span became resident on a worker, or it was
// evicted. The fak-owned fleet index consumes these; it does not reimplement the
// engine's cache (issue non-goal — consume KV-events, do not fork vLLM/SGLang).
type residencyOp string

const (
	// ResidentAdd folds a "worker now holds this prefix KV" signal into the index
	// (a vLLM block-stored event / an SGLang radix-insert / a native radixkv insert).
	ResidentAdd residencyOp = "resident"
	// ResidentDrop folds an eviction (cachemeta.KVOffload / an LRU leaf eviction) so
	// the index stops routing to a worker that no longer holds the span.
	ResidentDrop residencyOp = "evicted"
)

// KVCacheEvent is the fleet-index unit one residency transition lowers into. Prefix
// is the leading run of stable SEGMENT identities of the shared prefix whose KV the
// transition concerns — a token-block hash from a vLLM KV-cache-event, an SGLang
// radix segment, or a gateway-derived per-message digest (see prefixSegments). The
// index is segment-typed and engine-blind, so the same Ingest serves both tracks.
// This is the consumer the cachemeta.KVRoute seam (a KV-aware router pinning a
// request to the replica holding the span) was named-but-unconsumed for.
type KVCacheEvent struct {
	Worker string
	Op     residencyOp
	Prefix []string
}

// ResidentPrefixSource is the native (Track B) emitter seam: anything that can report
// the resident token-prefixes it currently holds. internal/radixkv (the shipped
// single-process radix prefix cache) is the native source — a per-worker adapter
// reads its resident leaf paths and feeds them in — but the gateway depends only on
// this interface, never on the cache implementation, so the dependency stays inverted.
type ResidentPrefixSource interface {
	// ResidentPrefixes returns the prefixes (each a leading segment run) the source
	// currently holds KV for, most-significant prefix first is not required.
	ResidentPrefixes() [][]string
}

// ResidentWorker is one worker's measured overlap with a queried prefix — the row the
// P/D orchestrator reads to co-locate decode with the replica holding the prefill KV.
type ResidentWorker struct {
	Worker  string
	Overlap int
}

// ResidencyView is the documented, read-only contract the prefill/decode orchestration
// seed (#28 native role split; #37 external P/D orchestration) consumes. It is THAT
// seed's named prerequisite: before splitting prefill from decode, the orchestrator
// must know which replica already holds (or, via a residency event, will receive) a
// request's KV, so it can place the decode where the prefill KV lives instead of
// moving bytes. PrefixResidencyIndex implements it; the orchestrator depends only on
// this interface, never on the index's internals.
type ResidencyView interface {
	// Overlap reports the longest resident leading-segment run worker is known to hold
	// for prefix (0 when the worker holds nothing of it).
	Overlap(worker string, prefix []string) int
	// ResidentWorkers lists every worker with nonzero overlap for prefix, best-overlap
	// first (ties broken by worker id) — the placement candidate set, ranked.
	ResidentWorkers(prefix []string) []ResidentWorker
}

// workerResidency is one worker's APPROXIMATE residency: a capacity-bounded LRU set of
// the prefixes it is known to hold KV for. Overlap is the longest common leading run
// over the held set — the SGLang-approximate per-worker tree, not an exact global
// state (issue non-goal: approximate per-worker trees + skew fallback are the target).
type workerResidency struct {
	capacity int
	order    []string            // LRU join-keys, least- → most-recently-used
	held     map[string][]string // join-key → prefix segments
}

func newWorkerResidency(capacity int) *workerResidency {
	if capacity < 1 {
		capacity = 1
	}
	return &workerResidency{capacity: capacity, held: make(map[string][]string, capacity)}
}

func residencyKey(prefix []string) string { return strings.Join(prefix, "\x1f") }

func (w *workerResidency) touch(k string) {
	for i, kk := range w.order {
		if kk == k {
			w.order = append(w.order[:i], w.order[i+1:]...)
			break
		}
	}
	w.order = append(w.order, k)
}

func (w *workerResidency) admit(prefix []string) {
	k := residencyKey(prefix)
	if _, ok := w.held[k]; ok {
		w.touch(k)
		return
	}
	if len(w.order) >= w.capacity {
		evict := w.order[0]
		w.order = w.order[1:]
		delete(w.held, evict)
	}
	w.order = append(w.order, k)
	w.held[k] = prefix
}

func (w *workerResidency) drop(prefix []string) {
	k := residencyKey(prefix)
	if _, ok := w.held[k]; !ok {
		return
	}
	delete(w.held, k)
	for i, kk := range w.order {
		if kk == k {
			w.order = append(w.order[:i], w.order[i+1:]...)
			break
		}
	}
}

func (w *workerResidency) overlap(prefix []string) int {
	best := 0
	for _, seg := range w.held {
		if l := segmentPrefixLen(seg, prefix); l > best {
			best = l
		}
	}
	return best
}

// segmentPrefixLen is the length of the shared leading run of two segment sequences.
func segmentPrefixLen(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// PrefixResidencyIndex is the fleet-level, per-worker prefix-residency index: a map
// from worker id to that worker's approximate resident-prefix set. It is the
// shared-substrate structure both emitters populate and the scorer reads. Safe for
// concurrent use.
type PrefixResidencyIndex struct {
	mu       sync.Mutex
	capacity int
	workers  map[string]*workerResidency
}

// NewPrefixResidencyIndex builds an empty index. capacityPerWorker bounds how many
// distinct resident prefixes each worker's approximate tree retains (LRU past it).
func NewPrefixResidencyIndex(capacityPerWorker int) *PrefixResidencyIndex {
	if capacityPerWorker < 1 {
		capacityPerWorker = 1
	}
	return &PrefixResidencyIndex{capacity: capacityPerWorker, workers: make(map[string]*workerResidency)}
}

func (x *PrefixResidencyIndex) workerLocked(id string) *workerResidency {
	w := x.workers[id]
	if w == nil {
		w = newWorkerResidency(x.capacity)
		x.workers[id] = w
	}
	return w
}

// Ingest folds one KV-cache-event (Track A) into the index. An empty worker or prefix
// is a no-op so a malformed event stream cannot corrupt the index.
func (x *PrefixResidencyIndex) Ingest(ev KVCacheEvent) {
	if ev.Worker == "" || len(ev.Prefix) == 0 {
		return
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	w := x.workerLocked(ev.Worker)
	if ev.Op == ResidentDrop {
		w.drop(ev.Prefix)
		return
	}
	w.admit(append([]string(nil), ev.Prefix...))
}

// Observe is the convenience emitter both the live scorer (after it homes a request)
// and the Track-B adapter use: it records that worker now holds prefix's KV.
func (x *PrefixResidencyIndex) Observe(worker string, prefix []string) {
	x.Ingest(KVCacheEvent{Worker: worker, Op: ResidentAdd, Prefix: prefix})
}

// IngestResidentPrefixes is the Track-B (NATIVE) fold: pull a worker's current
// resident prefixes from a native source (internal/radixkv) and admit each. Returns
// the count admitted. Wiring a single emitter is enough — the schema is shared.
func (x *PrefixResidencyIndex) IngestResidentPrefixes(worker string, src ResidentPrefixSource) int {
	if worker == "" || src == nil {
		return 0
	}
	n := 0
	for _, p := range src.ResidentPrefixes() {
		if len(p) == 0 {
			continue
		}
		x.Observe(worker, p)
		n++
	}
	return n
}

// Overlap implements ResidencyView.
func (x *PrefixResidencyIndex) Overlap(worker string, prefix []string) int {
	if worker == "" || len(prefix) == 0 {
		return 0
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if w := x.workers[worker]; w != nil {
		return w.overlap(prefix)
	}
	return 0
}

// Occupancy is a worker's resident distinct-prefix count — the bounded load term the
// scorer's cold placement uses to fill the fleet evenly before any worker evicts.
func (x *PrefixResidencyIndex) Occupancy(worker string) int {
	x.mu.Lock()
	defer x.mu.Unlock()
	if w := x.workers[worker]; w != nil {
		return len(w.order)
	}
	return 0
}

// ResidentWorkers implements ResidencyView.
func (x *PrefixResidencyIndex) ResidentWorkers(prefix []string) []ResidentWorker {
	if len(prefix) == 0 {
		return nil
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	out := make([]ResidentWorker, 0, len(x.workers))
	for id, w := range x.workers {
		if ov := w.overlap(prefix); ov > 0 {
			out = append(out, ResidentWorker{Worker: id, Overlap: ov})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Overlap != out[j].Overlap {
			return out[i].Overlap > out[j].Overlap
		}
		return out[i].Worker < out[j].Worker
	})
	return out
}

// SkewThreshold is the documented balancing threshold that decides when the per-worker
// trees have skewed far enough that locality routing would herd shared-prefix traffic
// onto one hot replica — at which point the scorer drops locality and routes purely by
// load (SGLang Router's balancing-threshold behavior). A fleet is "skewed" when BOTH
// hold: the busiest candidate's load exceeds the idlest by at least AbsLoad (absolute
// guard, so a near-balanced fleet is never tripped by noise) AND is at least RelLoad×
// the idlest (relative guard, so the threshold scales with fleet load). Rationale: the
// absolute term stops fallback from firing on tiny fleets where a 1-request gap is
// meaningless; the relative term keeps locality alive on a uniformly busy fleet where
// every worker is loaded but none is disproportionately hot.
type SkewThreshold struct {
	AbsLoad int
	RelLoad float64
}

// DefaultSkewThreshold is the standing threshold: a busiest-minus-idlest load gap of 8
// in-flight requests AND a busiest ≥ 1.5× the idlest. Documented so an operator can
// tune the locality/balance trade-off without reading the code.
func DefaultSkewThreshold() SkewThreshold { return SkewThreshold{AbsLoad: 8, RelLoad: 1.5} }

// CacheAwarePolicy is the power-of-two-choices, cache-aware pick policy. For each
// request it forms TWO choices over the live candidate set — the LOCALITY target (the
// worker with the longest resident prefix overlap) and the BALANCE target (the
// least-loaded worker) — and routes to whichever wins the cache-aware score
// (prefix-overlap-length × inverse-load). Two choices, picked by score, is the
// power-of-two-choices discipline that keeps a hot prefix's holder from absorbing all
// of its traffic: as the holder's load climbs, its score falls toward the idle balance
// target. Past the documented SkewThreshold it drops locality entirely and routes by
// load alone. Safe for concurrent use.
type CacheAwarePolicy struct {
	mu    sync.Mutex
	index *PrefixResidencyIndex
	skew  SkewThreshold
}

// NewCacheAwarePolicy builds the policy over a residency index (a fresh one is created
// when index is nil) and a skew threshold (defaults filled when zero).
func NewCacheAwarePolicy(index *PrefixResidencyIndex, skew SkewThreshold) *CacheAwarePolicy {
	if index == nil {
		index = NewPrefixResidencyIndex(64)
	}
	def := DefaultSkewThreshold()
	if skew.AbsLoad < 1 {
		skew.AbsLoad = def.AbsLoad
	}
	if skew.RelLoad < 1 {
		skew.RelLoad = def.RelLoad
	}
	return &CacheAwarePolicy{index: index, skew: skew}
}

// Index exposes the underlying residency index so callers can attach emitters (Ingest
// for Track A, IngestResidentPrefixes for Track B) and so the P/D seed can read it as
// a ResidencyView.
func (p *CacheAwarePolicy) Index() *PrefixResidencyIndex { return p.index }

// effectiveLoad is a worker's load for scoring: its resident occupancy (the bounded
// fill term that spreads cold prefixes to emptier workers) plus any external load
// (the live in-flight count membership supplies). ext may be nil.
func (p *CacheAwarePolicy) effectiveLoad(worker string, ext func(string) int) int {
	l := p.index.Occupancy(worker)
	if ext != nil {
		l += ext(worker)
	}
	return l
}

// score is the cache-aware scorer: prefix-overlap-length × inverse-load, with
// inverse-load = 1/(1+load) so a zero-load worker scores its full overlap and load
// only ever discounts. A cold worker (overlap 0) scores 0.
func (p *CacheAwarePolicy) score(worker string, prefix []string, ext func(string) int) float64 {
	ov := float64(p.index.Overlap(worker, prefix))
	return ov / float64(1+p.effectiveLoad(worker, ext))
}

// pickWorker is the name-level core the PlannerReplica adapter and the measurement
// harness share. It returns the chosen worker, whether that worker already held the
// FULL prefix (a routing-decision cache hit — no cold re-prefill), and ok=false only
// when there are no workers. It self-populates the index (Observe) so native routing
// builds residency the way an emitter would.
func (p *CacheAwarePolicy) pickWorker(workers []string, prefix []string, ext func(string) int) (chosen string, hit bool, ok bool) {
	if len(workers) == 0 {
		return "", false, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	loc := p.localityTarget(workers, prefix, ext)
	bal := p.balanceTarget(workers, ext)

	maxL, minL := p.loadRange(workers, ext)
	skewed := maxL-minL >= p.skew.AbsLoad && float64(maxL) >= p.skew.RelLoad*float64(maxInt(minL, 1))

	pick := loc
	if skewed {
		// Balancing threshold crossed: drop locality, route by load alone so
		// shared-prefix traffic stops herding onto the hot holder.
		pick = bal
	} else if p.score(bal, prefix, ext) > p.score(loc, prefix, ext) {
		pick = bal
	}

	hit = p.index.Overlap(pick, prefix) == len(prefix)
	p.index.Observe(pick, prefix)
	return pick, hit, true
}

// localityTarget is the worker holding the longest resident overlap with prefix; ties
// break to the lighter-loaded worker, then to the lower worker id (deterministic).
func (p *CacheAwarePolicy) localityTarget(workers []string, prefix []string, ext func(string) int) string {
	best := workers[0]
	bestOv, bestLoad := p.index.Overlap(best, prefix), p.effectiveLoad(best, ext)
	for _, w := range workers[1:] {
		ov, load := p.index.Overlap(w, prefix), p.effectiveLoad(w, ext)
		switch {
		case ov > bestOv:
			best, bestOv, bestLoad = w, ov, load
		case ov == bestOv && load < bestLoad:
			best, bestLoad = w, load
		case ov == bestOv && load == bestLoad && w < best:
			best = w
		}
	}
	return best
}

// balanceTarget is the least-loaded worker; ties break to the lower worker id.
func (p *CacheAwarePolicy) balanceTarget(workers []string, ext func(string) int) string {
	best := workers[0]
	bestLoad := p.effectiveLoad(best, ext)
	for _, w := range workers[1:] {
		if load := p.effectiveLoad(w, ext); load < bestLoad || (load == bestLoad && w < best) {
			best, bestLoad = w, load
		}
	}
	return best
}

func (p *CacheAwarePolicy) loadRange(workers []string, ext func(string) int) (max, min int) {
	max, min = p.effectiveLoad(workers[0], ext), p.effectiveLoad(workers[0], ext)
	for _, w := range workers[1:] {
		l := p.effectiveLoad(w, ext)
		if l > max {
			max = l
		}
		if l < min {
			min = l
		}
	}
	return max, min
}

// Pick implements PickPolicy: it adapts the live PlannerReplica candidate set to the
// name-level core and maps the chosen name back to its replica. ok=false only when the
// candidate set is empty (the router then falls back to its round-robin path).
func (p *CacheAwarePolicy) Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool) {
	if len(candidates) == 0 {
		return PlannerReplica{}, false
	}
	names := make([]string, len(candidates))
	byName := make(map[string]PlannerReplica, len(candidates))
	for i, c := range candidates {
		names[i] = c.Name
		byName[c.Name] = c
	}
	chosen, _, ok := p.pickWorker(names, prefix, load)
	if !ok {
		return PlannerReplica{}, false
	}
	return byName[chosen], true
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CacheAwareRoutingResult is the measured witness for the cache-aware-beats-round-robin
// acceptance: both policies' prefix-cache hit-rates on the SAME synthetic shared-prefix
// stream, over an index with identical residency mechanics, so the lift isolates the
// PICK policy exactly (the fleet analogue of the on-instance FCFS→cache-aware recovery).
type CacheAwareRoutingResult struct {
	Workers           int                 `json:"workers"`
	Workload          KVFleetWorkloadSpec `json:"workload"`
	Requests          int                 `json:"requests"`
	RoundRobinHitRate float64             `json:"round_robin_hit_rate"`
	CacheAwareHitRate float64             `json:"cache_aware_p2c_hit_rate"`
	HitRateLift       float64             `json:"hit_rate_lift_x"`
}

// MeasureCacheAwareRouting routes one shared-prefix workload through round-robin and
// through the cache-aware power-of-two policy and reports the measured hit-rates. Pure
// and deterministic: the same spec always yields the same numbers, so a committed
// witness stays reproducible. Both arms use a PrefixResidencyIndex of the same
// capacity; only the placement differs, so the lift is the policy's, not the cache's.
func MeasureCacheAwareRouting(spec KVFleetWorkloadSpec) CacheAwareRoutingResult {
	stream := spec.buildStream()
	n := spec.Instances
	if n < 1 {
		n = 1
	}
	names := make([]string, n)
	for i := range names {
		names[i] = "w" + strconv.Itoa(i)
	}

	// Baseline: cache-blind round-robin over identical residency mechanics. A hit is a
	// request whose round-robin-chosen worker already held the full prefix.
	rrIdx := NewPrefixResidencyIndex(spec.CapPerInstance)
	rrHits, cursor := 0, 0
	for _, key := range stream {
		prefix := []string{key}
		chosen := names[cursor%n]
		cursor++
		if rrIdx.Overlap(chosen, prefix) == len(prefix) {
			rrHits++
		}
		rrIdx.Observe(chosen, prefix)
	}

	// Cache-aware power-of-two policy on the same stream and worker set.
	policy := NewCacheAwarePolicy(NewPrefixResidencyIndex(spec.CapPerInstance), DefaultSkewThreshold())
	caHits := 0
	for _, key := range stream {
		if _, hit, ok := policy.pickWorker(names, []string{key}, nil); ok && hit {
			caHits++
		}
	}

	res := CacheAwareRoutingResult{
		Workers:  n,
		Workload: spec,
		Requests: len(stream),
	}
	if len(stream) > 0 {
		res.RoundRobinHitRate = float64(rrHits) / float64(len(stream))
		res.CacheAwareHitRate = float64(caHits) / float64(len(stream))
	}
	if res.RoundRobinHitRate > 0 {
		res.HitRateLift = res.CacheAwareHitRate / res.RoundRobinHitRate
	}
	return res
}
