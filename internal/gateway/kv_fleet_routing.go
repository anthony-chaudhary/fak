package gateway

import "strconv"

// KV-aware fleet routing — the cross-instance analogue of fak's on-instance
// cache-aware scheduling (the FCFS 62.1% -> 86.7% radix-hit recovery folded into
// the kv-hit-rate parity row). The on-instance scheduler reorders the queue of
// ONE kernel so a request runs while its prefix KV is resident; this routes a
// request ACROSS a fleet of kernels to the instance that already holds its prefix
// KV, so the prefill is reused instead of recomputed cold on whichever worker a
// cache-blind balancer happened to pick.
//
// It is the fleet-level request-to-cached-block router the `kv-cache-aware-routing`
// industry-scorecard gap names: fak had cache-aware SCHEDULING within one instance
// and a coherence bus, but no cross-GPU residency router — so a request could land
// on a worker that must re-prefill a prefix another worker already holds. The
// router below carries the same residency signal across instances. A cold prefix
// lands on the least-loaded instance and is kept balanced, the overlap-minus-load
// placement the Dynamo Smart TierPolicy / SGLang cache-aware router approximate.
//
// What is measured here is the routing DECISION hit-rate — does the router send a
// request to an instance that already holds its prefix? — the same kind of number
// as the on-instance radix hit-rate. The downstream wall-clock TTFT/throughput win
// those hits produce on real GPUs (Baseten/Dynamo's +62% output tok/s, GORGO's
// 2.5x TTFT) is host-gated and not claimed here.

// residentSet is one instance's bounded set of resident prefix-KV blocks: an LRU
// keyed by prefix identity. capacity is the number of DISTINCT prefixes whose KV
// fits in that instance's budget; admitting past capacity evicts the LRU prefix.
type residentSet struct {
	capacity int
	order    []string // least-recently-used first, most-recently-used last
	held     map[string]struct{}
}

func newResidentSet(capacity int) *residentSet {
	if capacity < 1 {
		capacity = 1
	}
	return &residentSet{capacity: capacity, held: make(map[string]struct{}, capacity)}
}

func (s *residentSet) holds(key string) bool {
	_, ok := s.held[key]
	return ok
}

// touch marks an already-resident key most-recently-used (a hit re-prefills nothing).
func (s *residentSet) touch(key string) {
	for i, k := range s.order {
		if k == key {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	s.order = append(s.order, key)
}

// admit brings a cold prefix resident, evicting the LRU prefix if at capacity.
func (s *residentSet) admit(key string) {
	if s.holds(key) {
		s.touch(key)
		return
	}
	if len(s.order) >= s.capacity {
		evict := s.order[0]
		s.order = s.order[1:]
		delete(s.held, evict)
	}
	s.order = append(s.order, key)
	s.held[key] = struct{}{}
}

// FleetRouter places a request (identified by the shareable prefix it would
// re-prefill cold) on one instance of a fixed fleet and reports whether that
// instance already held the prefix KV — a cross-instance hit, no re-prefill.
type FleetRouter interface {
	Route(prefixKey string) (instance int, hit bool)
}

// CacheBlindRouter is the baseline: round-robin placement that ignores residency
// (what the existing ReplicaDispatch.pick does). It still records whether the chosen
// instance happened to hold the prefix, so its hit-rate is directly comparable to
// the KV-aware router's on the same request stream.
type CacheBlindRouter struct {
	instances []*residentSet
	next      int
}

// newResidentInstances builds a clamped (n>=1) slice of n resident sets, each able to keep
// capPerInstance distinct prefixes resident — the shared fleet-construction step of the
// cache-blind and KV-aware routers.
func newResidentInstances(n, capPerInstance int) []*residentSet {
	if n < 1 {
		n = 1
	}
	insts := make([]*residentSet, n)
	for i := range insts {
		insts[i] = newResidentSet(capPerInstance)
	}
	return insts
}

// NewCacheBlindRouter builds a round-robin fleet of n instances, each able to keep
// capPerInstance distinct prefixes resident.
func NewCacheBlindRouter(n, capPerInstance int) *CacheBlindRouter {
	return &CacheBlindRouter{instances: newResidentInstances(n, capPerInstance)}
}

func (r *CacheBlindRouter) Route(prefixKey string) (int, bool) {
	i := r.next % len(r.instances)
	r.next++
	hit := r.instances[i].holds(prefixKey)
	r.instances[i].admit(prefixKey)
	return i, hit
}

// FleetCacheRouter is the KV-aware router: it routes to an instance that already
// holds the prefix KV (locality), and lands a cold prefix on the least-loaded
// instance so prefix homes stay balanced (the overlap-minus-load placement). It
// is the cross-instance composition of the on-instance residency signal — the
// fleet-level request-to-cached-block router the kv-cache-aware-routing gap names.
type FleetCacheRouter struct {
	instances []*residentSet
	load      []int

	// tolerance arms the bounded load-imbalance slack on the sticky path (the
	// microsoft/vidur slack-tolerant sticky scheduler borrow; see routeSlack).
	// Zero keeps the rigid sticky behavior the committed hit-rate witness pins;
	// a positive value sheds a sticky prefix whose home instance's load exceeds
	// tolerance x the fleet's minimum load.
	tolerance float64

	// home is the sticky record the slack path consults: prefix key -> the
	// instance the policy currently pins it to. Nil unless the slack is armed
	// (NewSlackTolerantFleetCacheRouter); the rigid path stays record-free.
	home map[string]int
}

// NewFleetCacheRouter builds a KV-aware fleet of n instances, each able to keep
// capPerInstance distinct prefixes resident.
func NewFleetCacheRouter(n, capPerInstance int) *FleetCacheRouter {
	insts := newResidentInstances(n, capPerInstance)
	return &FleetCacheRouter{instances: insts, load: make([]int, len(insts))}
}

// NewSlackTolerantFleetCacheRouter builds a KV-aware fleet of n instances with the
// bounded load-imbalance slack armed: a prefix's home instance keeps its traffic
// while its load stays within tolerance x the fleet's minimum load, and sheds to
// the least-loaded instance once it exceeds the bound. Adapted from
// microsoft/vidur's TolerantStickyLOPUncachedGlobalScheduler (source:
// vidur/scheduler/global_scheduler/tolerant_sticky_lop_uncached_global_scheduler.py
// @abae7f63aa857300f5cdc6f5e0d27860cd24721b, MIT; re-expressed in Go, no copied
// code). The slack only bites above one replica: with a single instance the home
// IS the minimum, so the bound can never be exceeded. tolerance <= 0 falls back
// to the rigid sticky router.
func NewSlackTolerantFleetCacheRouter(n, capPerInstance int, tolerance float64) *FleetCacheRouter {
	r := NewFleetCacheRouter(n, capPerInstance)
	if tolerance > 0 {
		r.tolerance = tolerance
		r.home = make(map[string]int)
	}
	return r
}

func (r *FleetCacheRouter) Route(prefixKey string) (int, bool) {
	if r.tolerance > 0 && len(r.instances) > 1 {
		return r.routeSlack(prefixKey)
	}
	for i, s := range r.instances {
		if s.holds(prefixKey) {
			s.touch(prefixKey)
			r.load[i]++
			return i, true
		}
	}
	pick := r.coldPlacement()
	r.instances[pick].admit(prefixKey)
	r.load[pick]++
	return pick, false
}

// routeSlack is the slack-tolerant sticky placement. The prefix's home instance
// (the sticky record, seeded by its last placement) keeps the turn while its load
// stays within the bounded slack — load[home] <= tolerance x min(load) — because
// each turn there reuses resident prefix KV instead of a cold prefill. Past the
// bound the turn sheds to the least-loaded OTHER instance (a cold placement that
// duplicates the prefix's residency there) and the sticky record migrates with
// it, so the session tracks the cool side of the fleet until loads rebalance. A
// home that evicted the prefix (LRU pressure) loses the record and re-homes cold.
// Deterministic: ties break to the lower instance index, unlike the upstream
// random tie-break, so the witness is reproducible.
func (r *FleetCacheRouter) routeSlack(prefixKey string) (int, bool) {
	if r.home == nil {
		r.home = make(map[string]int)
	}
	home, pinned := r.home[prefixKey]
	liveHome := pinned && r.instances[home].holds(prefixKey)
	if liveHome && float64(r.load[home]) <= r.tolerance*float64(r.minLoad()) {
		// Within the bounded slack: keep the sticky placement (prefix hit).
		r.instances[home].touch(prefixKey)
		r.load[home]++
		return home, true
	}
	// No live sticky record (or the home evicted the prefix, or the home's load
	// exceeded the slack): place on the least-loaded instance, skipping the
	// overloaded home so a shed actually moves the session, and migrate the
	// sticky record with it.
	exclude := -1
	if liveHome {
		exclude = home
	}
	pick := r.leastLoaded(exclude)
	hit := r.instances[pick].holds(prefixKey)
	r.instances[pick].admit(prefixKey)
	r.load[pick]++
	r.home[prefixKey] = pick
	return pick, hit
}

// minLoad is the fleet's minimum in-flight request count — the slack bound's
// denominator (the upstream min_load_imbalance read over the replica set).
func (r *FleetCacheRouter) minLoad() int {
	m := r.load[0]
	for _, l := range r.load[1:] {
		if l < m {
			m = l
		}
	}
	return m
}

// leastLoaded picks the instance with the fewest in-flight requests, skipping
// except (the sticky home being shed from, so a shed always moves the session);
// ties break to the lower index. except may be -1 for no exclusion.
func (r *FleetCacheRouter) leastLoaded(except int) int {
	best := -1
	for i := range r.instances {
		if i == except {
			continue
		}
		if best == -1 || r.load[i] < r.load[best] {
			best = i
		}
	}
	return best
}

// coldPlacement homes a brand-new prefix on the instance with the most FREE cache
// (fewest resident distinct prefixes), so the fleet's aggregate KV is filled
// before anything is evicted — the overlap-minus-load rule with occupancy as the
// load term. Ties break to the lower request count, then the lower index, so the
// placement is deterministic.
func (r *FleetCacheRouter) coldPlacement() int {
	best := 0
	for i := 1; i < len(r.instances); i++ {
		bi, ci := len(r.instances[best].order), len(r.instances[i].order)
		switch {
		case ci < bi:
			best = i
		case ci == bi && r.load[i] < r.load[best]:
			best = i
		}
	}
	return best
}

// FleetRoutingStats is one routing policy's outcome over a request stream.
type FleetRoutingStats struct {
	Requests int     `json:"requests"`
	Hits     int     `json:"hits"`
	HitRate  float64 `json:"hit_rate"`
}

func runFleetPolicy(router FleetRouter, stream []string) FleetRoutingStats {
	hits := 0
	for _, key := range stream {
		if _, hit := router.Route(key); hit {
			hits++
		}
	}
	st := FleetRoutingStats{Requests: len(stream), Hits: hits}
	if len(stream) > 0 {
		st.HitRate = float64(hits) / float64(len(stream))
	}
	return st
}

// KVFleetWorkloadSpec describes the synthetic shared-prefix agent fleet the
// harness routes. It is deliberately in the realistic regime that motivates
// KV-aware routing: the aggregate distinct prefix working set (PrefixFamilies)
// EXCEEDS one fleet's resident capacity (Instances * CapPerInstance), so no
// router can hold everything and locality is what keeps the hot prefixes home.
// Traffic is Zipf-skewed (a few shared system-prompts / agent scaffolds dominate),
// as a real cross-agent fleet's is, and requests are interleaved one-per-family
// per round so each family's turns are spread across the stream (concurrent
// agents, not back-to-back bursts).
type KVFleetWorkloadSpec struct {
	Instances      int `json:"instances"`
	CapPerInstance int `json:"cap_per_instance_prefixes"`
	PrefixFamilies int `json:"prefix_families"`
	HotWeight      int `json:"zipf_hot_weight"`
}

// buildStream renders the spec into a deterministic request stream: family i
// (1-indexed) gets max(1, HotWeight/i) requests (a Zipf 1/i popularity), and the
// requests are dealt one-family-per-round so same-family turns are spread out.
func (w KVFleetWorkloadSpec) buildStream() []string {
	remaining := make([]int, w.PrefixFamilies)
	total := 0
	for i := range remaining {
		n := w.HotWeight / (i + 1)
		if n < 1 {
			n = 1
		}
		remaining[i] = n
		total += n
	}
	stream := make([]string, 0, total)
	for emitted := 0; emitted < total; {
		for i := range remaining {
			if remaining[i] > 0 {
				stream = append(stream, prefixKey(i))
				remaining[i]--
				emitted++
			}
		}
	}
	return stream
}

func prefixKey(family int) string {
	// A stable per-family prefix identity (the shared system-prompt / scaffold a
	// router would re-prefill cold). Content is irrelevant; identity is the key.
	return "prefix-family-" + strconv.Itoa(family)
}

// KVFleetCompetitor records the SOTA cross-replica cache-hit bar the fak number
// is read against — the same metric, so the comparison is apples-to-apples.
type KVFleetCompetitor struct {
	Name             string  `json:"name"`
	CrossReplicaHit  float64 `json:"cross_replica_cache_hit"`
	Replicas         int     `json:"replicas"`
	DownstreamEffect string  `json:"downstream_wallclock_effect"`
	Source           string  `json:"source"`
}

// KVFleetRoutingResult is the committed witness: both policies' hit-rates on the
// same stream, the lift, and the competitor bar. The cache_blind_round_robin arm
// is fak's own cache-blind baseline (what ReplicaDispatch does today), so the lift
// isolates KV-aware ROUTING the way the on-instance row isolated cache-aware
// SCHEDULING (FCFS 62.1% -> DFS 86.7%) on one held-constant kernel.
type KVFleetRoutingResult struct {
	Harness     string              `json:"harness"`
	Workload    KVFleetWorkloadSpec `json:"workload"`
	CacheBlind  FleetRoutingStats   `json:"cache_blind_round_robin"`
	KVAware     FleetRoutingStats   `json:"kv_aware_locality"`
	HitRateLift float64             `json:"hit_rate_lift_x"`
	Competitor  KVFleetCompetitor   `json:"competitor"`
}

// MeasureKVAwareFleetRouting runs the cache-blind and KV-aware routers over one
// shared-prefix fleet workload and returns the comparison. Pure and deterministic:
// the same spec always yields the same numbers, so the committed artifact stays
// reproducible.
func MeasureKVAwareFleetRouting(spec KVFleetWorkloadSpec) KVFleetRoutingResult {
	stream := spec.buildStream()
	blind := runFleetPolicy(NewCacheBlindRouter(spec.Instances, spec.CapPerInstance), stream)
	aware := runFleetPolicy(NewFleetCacheRouter(spec.Instances, spec.CapPerInstance), stream)
	lift := 0.0
	if blind.HitRate > 0 {
		lift = aware.HitRate / blind.HitRate
	}
	return KVFleetRoutingResult{
		Harness:     "TestKVAwareFleetRoutingHitRate",
		Workload:    spec,
		CacheBlind:  blind,
		KVAware:     aware,
		HitRateLift: lift,
		Competitor: KVFleetCompetitor{
			Name:             "Baseten on NVIDIA Dynamo (KV-aware router)",
			CrossReplicaHit:  0.89,
			Replicas:         4,
			DownstreamEffect: "-50% TTFT / +62% output tok/s / +61% RPS on Qwen3 480B ~50k-token inputs (host-gated; not measured here)",
			Source:           "https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/ (2025-01)",
		},
	}
}

// DefaultKVFleetWorkload is the harness's standing workload: 4 instances (matching
// Baseten's 4 replicas), 8 resident prefixes each (fleet capacity 32), against 28
// distinct prefix-families — the standard KV-aware-routing regime where the working
// set fits the FLEET (28 <= 32) but NOT one instance (28 > 8). So one cache-blind
// instance must thrash its 8 slots over all 28 families, while locality routing
// keeps each family resident on a stable home (7 per instance) and re-prefills
// nothing on a recurrence.
var DefaultKVFleetWorkload = KVFleetWorkloadSpec{
	Instances:      4,
	CapPerInstance: 8,
	PrefixFamilies: 28,
	HotWeight:      120,
}
